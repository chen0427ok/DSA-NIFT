"""
L3（第二步）— 情感知識圖譜 × Claude API 可控生成資料增強。

讀 outputs/l3_graph.pkl（L3 的圖）→ 針對「訓練資料代表性不足、尤其是高/低 arousal 兩端」
的 VA 目標區間，從圖上撈情緒種子詞 → 請 Claude 生成符合該情緒強度的第一人稱新住民反思短文
→ 標上目標 VA（含小幅 jitter）→ 輸出 data/train_aug.csv。

直接攻擊我們最痛的 arousal 分布壓縮問題：靠合成資料把高/低喚醒的樣本補回來。

用法（兩家 LLM 擇一；本地或 Colab 都能跑，純 API 不需 GPU）：
    # Claude（預設）
    export ANTHROPIC_API_KEY=sk-ant-...
    python augment_generate.py --per_bin 40

    # OpenAI
    export OPENAI_API_KEY=sk-...
    python augment_generate.py --provider openai --per_bin 40

    python augment_generate.py --dry_run     # 不呼叫 API，只印種子詞與 prompt（驗證用）
    python augment_generate.py --append      # 生成後併進 data/train.csv（先備份 train_base.csv）

需要：pip install anthropic（或 openai）pydantic；對應的 *_API_KEY 環境變數。
"""
import os
import csv
import pickle
import random
import argparse

from affective_graph import (build_graph, seeds_for_target, seeds_random,
                             seeds_by_expansion, seed_coherence)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")


def load_dotenv(path=None):
    """從 .env 讀 API key（不覆蓋已存在的環境變數）。預設找專案上層的 .env。"""
    path = path or os.path.join(HERE, "..", ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# 風格錨定（style anchoring）：用官方【無標籤】文本當生成的語域範例。
# 動機：實驗 5b 診斷出 A_MAE 的傷害來自合成文本的語域/風格偏移，而非標籤。
# ⚠️ 只使用官方公開的「文本」，不使用任何標籤（官方也未釋出標籤）。論文須明確聲明。
# ---------------------------------------------------------------------------

def load_unlabeled_texts(paths):
    """讀官方無標籤 csv（ID,Text 或 id,text），回傳 list[str]。"""
    texts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  ⚠ 找不到 {p}，略過")
            continue
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                low = {(k or "").strip().lower(): v for k, v in r.items()}
                t = (low.get("text") or "").strip()
                if t:
                    texts.append(t)
    return texts


def retrieve_style_examples(texts, seed_words, k=4, rng=None, mode="lexical"):
    """依種子詞從無標籤文本中檢索風格範例。

    mode="lexical"：計算每篇文本命中幾個種子詞，取最高分的 k 篇（同分隨機打散）。
                    這讓「情緒區間」與「風格範例」對齊——圖譜選出的情緒詞，
                    帶出的是目標域中真的在談那種情緒的段落。
    mode="random"： 純隨機抽 k 篇（消融用：檢驗「檢索」是否比「隨便給範例」好）。
    """
    rng = rng or random.Random(0)
    if not texts:
        return []
    if mode == "random" or not seed_words:
        return rng.sample(texts, min(k, len(texts)))
    scored = []
    for t in texts:
        hits = sum(1 for w in seed_words if w in t)
        scored.append((hits, rng.random(), t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = [t for h, _, t in scored[:k] if h > 0]
    if len(top) < k:  # 命中不足時用隨機補滿，避免 prompt 沒有風格範例
        pool = [t for _, _, t in scored[k:]]
        top += rng.sample(pool, min(k - len(top), len(pool)))
    return top
# 各 provider 的預設模型（可用 --model 覆寫）
DEFAULT_MODELS = {"anthropic": "claude-opus-4-8", "openai": "gpt-4o"}

# 目標 VA bins（1–9 量尺）。重點補 arousal 高/低兩端（瓶頸），valence 各區都鋪。
# (v_center, a_center, v_range, a_range, 情緒描述)
TARGET_BINS = [
    (3.0, 7.5, (1.0, 4.0), (7.0, 9.0), "負面效價、非常高喚醒（焦慮、憤怒、恐慌、崩潰）"),
    (5.0, 7.5, (4.0, 6.0), (7.0, 9.0), "中性效價、高喚醒（緊張、激動、震驚、坐立難安）"),
    (7.0, 7.5, (6.0, 9.0), (7.0, 9.0), "正面效價、高喚醒（興奮、雀躍、滿懷期待、感動到發抖）"),
    (3.0, 2.5, (1.0, 4.0), (1.0, 3.5), "負面效價、低喚醒（沮喪、疲憊、無力、提不起勁）"),
    (7.0, 2.5, (6.0, 9.0), (1.0, 3.5), "正面效價、低喚醒（平靜、滿足、安心、放鬆）"),
]


def load_graph():
    """優先讀 L3 已存的圖；沒有就臨時建（需要 outputs/l2_word_va.pkl）。"""
    p = os.path.join(OUT, "l3_graph.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    print("（找不到 outputs/l3_graph.pkl，改用 build_graph 臨時建圖）")
    G, _ = build_graph(k=8)
    return G


# 長度指令。fixed = 實驗 5 原始寫法（凍結，條件 C/E/F 用）。
# match_real = 依 ValidationSet 實測分布（中位 64、5–95 百分位 43–142、最長 226）重寫，
#   目的是解除「每段 50–120 字」把合成文本長度鎖死在 58–113 的問題（條件 F2）。
LENGTH_FIXED = "每段 50–120 字"
LENGTH_MATCH_REAL = (
    "每段長度要明顯有長有短，不要每段都差不多長：大約六成落在 50–90 字，"
    "約一成很短（20–45 字，像隨手記一句）、約一成偏長（120–200 字，寫得比較細）"
)
LENGTH_SPECS = {"fixed": LENGTH_FIXED, "match_real": LENGTH_MATCH_REAL}


def build_prompt(seed_words, desc, n, style_examples=None, length_spec=LENGTH_FIXED):
    """組生成 prompt。

    ⚠️ 相容性：不給 style_examples 且 seed_words 非空時，輸出與實驗 5 的原始 prompt
    【逐字相同】，確保條件 C 可復現。新增的區塊只在對應旗標開啟時才出現。
    """
    seeds = "、".join(seed_words) if seed_words else "（圖上無此區間種子詞，請自行掌握情緒強度）"
    style_block = ""
    if style_examples:
        ex = "\n".join(f"  - {t}" for t in style_examples)
        style_block = (
            f"【風格範例】以下是真實的新住民文本，請模仿它們的口吻、句長與用詞習慣，"
            f"但**內容必須完全不同、不可改寫或照抄**：\n{ex}\n"
        )
    return (
        f"你要協助建立中文情緒分析訓練資料。請以「新住民（嫁來/移居台灣的外籍配偶或移工）」的第一人稱，"
        f"寫 {n} 段彼此獨立、內容各異的生活反思短文。\n\n"
        f"【情緒要求】每段都要明確傳達：{desc}。情緒強度要到位、可從文字明顯感受到。\n"
        f"【可用情緒詞】可自然帶入這些詞當情緒參考（不必全用、不要硬塞；"
        f"凡與新住民日常語境不符、過於暴力或粗俗的詞請直接忽略）：{seeds}\n"
        f"{style_block}"
        f"【內容要求】題材多元：工作、家庭、語言、孩子教育、思鄉、人際、就醫、證件/身分、節慶等都可；"
        f"{length_spec}；用第一人稱「我」；像真人日記/自述，不要像新聞或教科書；不要編號、不要標題。\n"
        f"只回傳這 {n} 段短文。"
    )


def build_prompt_no_seeds(desc, n, style_examples=None, length_spec=LENGTH_FIXED):
    """條件 N：完全不給種子詞（連【可用情緒詞】區塊都拿掉），只給情緒描述。

    對照組的意義：如果 N 與 C 打平，代表 LLM 光看情緒描述就能寫出對的 arousal，
    整個詞典/圖譜的種子詞機制都是多餘的——這是最基本、也最該先排除的可能。
    """
    style_block = ""
    if style_examples:
        ex = "\n".join(f"  - {t}" for t in style_examples)
        style_block = (
            f"【風格範例】以下是真實的新住民文本，請模仿它們的口吻、句長與用詞習慣，"
            f"但**內容必須完全不同、不可改寫或照抄**：\n{ex}\n"
        )
    return (
        f"你要協助建立中文情緒分析訓練資料。請以「新住民（嫁來/移居台灣的外籍配偶或移工）」的第一人稱，"
        f"寫 {n} 段彼此獨立、內容各異的生活反思短文。\n\n"
        f"【情緒要求】每段都要明確傳達：{desc}。情緒強度要到位、可從文字明顯感受到。\n"
        f"{style_block}"
        f"【內容要求】題材多元：工作、家庭、語言、孩子教育、思鄉、人際、就醫、證件/身分、節慶等都可；"
        f"{length_spec}；用第一人稱「我」；像真人日記/自述，不要像新聞或教科書；不要編號、不要標題。\n"
        f"只回傳這 {n} 段短文。"
    )




CSV_FIELDS = ["id", "granularity", "text", "valence", "arousal"]


def _write_rows(path, rows):
    """把目前累積的 rows 寫出（逐 bin 落檔用，覆寫式）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def pick_seeds(G, mode, vr, ar, n, rng, n_anchors=4, hops=2):
    """依消融條件選種子詞。回傳 [(word, v, a)]。"""
    if mode == "none":
        return []
    if mode == "random":
        return seeds_random(G, n=n, rng=rng)
    if mode == "lookup":                      # 實驗 5 原始行為（凍結）
        return seeds_for_target(G, v_range=vr, a_range=ar, n=n)
    if mode == "graph":
        return seeds_by_expansion(G, v_range=vr, a_range=ar, n=n, rng=rng,
                                  n_anchors=n_anchors, hops=hops)
    raise ValueError(f"未知 seed_mode: {mode}")


def make_client(provider):
    """建立 LLM client（讀對應的環境變數 API key）。"""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()          # 讀 ANTHROPIC_API_KEY
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI()                        # 讀 OPENAI_API_KEY
    raise ValueError(f"未知 provider: {provider}")


def _with_retry(fn, tries=9, base=4.0, label=""):
    """指數退避重試。API 529 Overloaded / 429 rate limit / 連線錯誤都是暫時性的，
    不重試的話一次抖動就會讓整批數百篇生成全部作廢（實測發生過）。"""
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            name = type(e).__name__
            transient = any(s in name for s in ("Overloaded", "RateLimit", "APIConnection",
                                                "InternalServer", "Timeout"))
            if not transient or i == tries - 1:
                raise
            wait = base * (2 ** i) + random.uniform(0, 2)
            print(f"  ⚠ {name}{(' ' + label) if label else ''}，{wait:.0f}s 後重試 "
                  f"({i+1}/{tries-1})", flush=True)
            time.sleep(wait)


def generate_bin(provider, client, model, ParsedModel, seed_words, desc, n,
                 style_examples=None, seed_mode="lookup", length_spec=LENGTH_FIXED):
    """呼叫 LLM 生成 n 段短文，回傳 list[str]。兩家都用 structured output 拿乾淨陣列。"""
    if seed_mode == "none":
        prompt = build_prompt_no_seeds(desc, n, style_examples, length_spec)
    else:
        prompt = build_prompt(seed_words, desc, n, style_examples, length_spec)
    if provider == "anthropic":
        resp = _with_retry(lambda: client.messages.parse(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
            output_format=ParsedModel,
        ))
        texts = resp.parsed_output.texts
    else:  # openai
        resp = _with_retry(lambda: client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ParsedModel,
        ))
        texts = resp.choices[0].message.parsed.texts
    return [t.strip() for t in texts if t and t.strip()]


def _tokset(text):
    """jieba 斷詞後的詞集合（去單字標點），用來算近似相似度。"""
    import jieba
    return {w for w in jieba.cut(text) if len(w.strip()) > 1}


def _too_similar(ts, accepted, thr):
    """ts 與任何已接受文本的 Jaccard 相似度 > thr 就算重複。"""
    for prev in accepted:
        union = ts | prev
        if union and len(ts & prev) / len(union) > thr:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_bin", type=int, default=40, help="每個 VA 區間要生幾篇")
    ap.add_argument("--chunk", type=int, default=20, help="每次 API 呼叫請幾篇（湊滿 per_bin 會多次呼叫）")
    ap.add_argument("--jitter", type=float, default=0.4, help="VA 標籤在中心值附近的隨機抖動幅度")
    ap.add_argument("--seeds_per_bin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup_threshold", type=float, default=0.5,
                    help="近似去重門檻（Jaccard 詞集相似度 > 此值就丟棄）")
    ap.add_argument("--no_dedup", action="store_true", help="關閉去重")
    ap.add_argument("--max_calls_per_bin", type=int, default=12,
                    help="每區間最多呼叫幾次 API（去重丟太多時的安全上限）")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None, help="覆寫模型；預設依 provider（claude-opus-4-8 / gpt-4o）")
    ap.add_argument("--out", default=os.path.join(DATA, "train_aug.csv"))
    ap.add_argument("--append", action="store_true", help="生成後併進 data/train.csv（先備份 train_base.csv）")
    ap.add_argument("--dry_run", action="store_true", help="不呼叫 API，只印種子詞與 prompt")
    # ---- 消融旋鈕（見 docs/paper/kg_experiment_plan.md）----
    ap.add_argument("--seed_mode", choices=["none", "random", "lookup", "graph"], default="lookup",
                    help="種子詞策略：none=條件N（不給詞）/ random=條件A / "
                         "lookup=條件C（實驗5原始行為，預設）/ graph=條件E（圖擴散 G1+G3）")
    ap.add_argument("--style_anchor", action="append", default=[],
                    help="風格錨定用的官方無標籤 csv（可重複指定；如 ../DSANIDF_TestSet.csv）")
    ap.add_argument("--n_style", type=int, default=4, help="每次呼叫附幾篇風格範例")
    ap.add_argument("--style_mode", choices=["lexical", "random"], default="lexical",
                    help="風格範例檢索方式：lexical=依種子詞命中檢索 / random=隨機抽（消融）")
    ap.add_argument("--length", choices=["fixed", "match_real"], default="fixed",
                    help="長度指令：fixed=每段50-120字（實驗5原始，預設）/ "
                         "match_real=依 ValidationSet 實測分布，明確要求長短不一（條件 F2）")
    ap.add_argument("--n_anchors", type=int, default=4, help="graph 模式：每次抽幾個 anchor")
    ap.add_argument("--hops", type=int, default=2, help="graph 模式：沿邊擴散幾跳")
    args = ap.parse_args()
    load_dotenv()
    random.seed(args.seed)
    rng = random.Random(args.seed)
    model = args.model or DEFAULT_MODELS[args.provider]

    length_spec = LENGTH_SPECS[args.length]

    G = load_graph()

    style_texts = []
    if args.style_anchor:
        style_texts = load_unlabeled_texts(args.style_anchor)
        print(f"風格錨定：載入 {len(style_texts)} 篇官方無標籤文本"
              f"（僅用文本，未使用任何標籤）")
    print(f"seed_mode={args.seed_mode}  style_anchor={'ON' if style_texts else 'OFF'}"
          f"  style_mode={args.style_mode}")

    client = ParsedModel = None
    if not args.dry_run:
        from pydantic import BaseModel

        class GeneratedTexts(BaseModel):
            texts: list[str]

        ParsedModel = GeneratedTexts
        client = make_client(args.provider)
        print(f"provider={args.provider}  model={model}")

    rows = []
    accepted = []   # 跨所有 bin 的已接受文本詞集，連跨區間重複也擋
    dropped_total = 0
    for i, (vc, ac, vr, ar, desc) in enumerate(TARGET_BINS):
        seeds = pick_seeds(G, args.seed_mode, vr, ar, args.seeds_per_bin, rng,
                           args.n_anchors, args.hops)
        seed_words = [w for w, _, _ in seeds]
        print(f"\n[bin {i+1}/{len(TARGET_BINS)}] V≈{vc} A≈{ac}  {desc}")
        print(f"  種子詞({len(seed_words)}): {'、'.join(seed_words) or '（無 — 條件 N）'}")
        if seed_words:
            print(f"  語意連貫度 (圖上相連詞對比例): {seed_coherence(G, seed_words):.3f}")

        if args.dry_run:
            ex = retrieve_style_examples(style_texts, seed_words, args.n_style, rng,
                                         args.style_mode) if style_texts else []
            print("  --- prompt 預覽 ---")
            p = (build_prompt_no_seeds(desc, min(args.chunk, args.per_bin), ex, length_spec)
                 if args.seed_mode == "none"
                 else build_prompt(seed_words, desc, min(args.chunk, args.per_bin), ex, length_spec))
            print("  " + p.replace("\n", "\n  "))
            continue

        texts, calls = [], 0
        while len(texts) < args.per_bin and calls < args.max_calls_per_bin:
            need = min(args.chunk, args.per_bin - len(texts))
            # G3：每次呼叫都重抽 anchor → 種子詞不同 → 解決 80 篇共用一組詞的問題
            if args.seed_mode == "graph" and calls > 0:
                seeds = pick_seeds(G, args.seed_mode, vr, ar, args.seeds_per_bin, rng,
                                   args.n_anchors, args.hops)
                seed_words = [w for w, _, _ in seeds]
            ex = retrieve_style_examples(style_texts, seed_words, args.n_style, rng,
                                         args.style_mode) if style_texts else []
            got = generate_bin(args.provider, client, model, ParsedModel, seed_words, desc, need,
                               style_examples=ex, seed_mode=args.seed_mode,
                               length_spec=length_spec)
            calls += 1
            if not got:
                print("  ⚠ 這批回傳 0 篇，跳出避免無限迴圈"); break
            for t in got:
                if not args.no_dedup:
                    ts = _tokset(t)
                    if _too_similar(ts, accepted, args.dedup_threshold):
                        dropped_total += 1; continue
                    accepted.append(ts)
                texts.append(t)
            print(f"  已接受 {len(texts)}/{args.per_bin}（呼叫 {calls} 次，累計去重丟棄 {dropped_total}）")
        if len(texts) < args.per_bin:
            print(f"  ⚠ 達呼叫上限仍只湊到 {len(texts)} 篇（去重太嚴可放寬 --dedup_threshold 或加 --max_calls_per_bin）")

        for t in texts[:args.per_bin]:
            v = round(min(9.0, max(1.0, vc + random.uniform(-args.jitter, args.jitter))), 2)
            a = round(min(9.0, max(1.0, ac + random.uniform(-args.jitter, args.jitter))), 2)
            rows.append({"id": f"AUG_{len(rows)+1:04d}", "granularity": "augment",
                         "text": t, "valence": v, "arousal": a})

        # 每個 bin 結束就先落檔：API 在最後一個 bin 掛掉時，不會賠掉前面幾百篇（實測發生過）
        _write_rows(args.out, rows)
        print(f"  ✓ 已存檔 {len(rows)} 篇 → {args.out}", flush=True)

    if args.dry_run:
        print(f"\n（dry_run）共 {len(TARGET_BINS)} 個 VA 區間，未呼叫 API、未輸出檔案。")
        return

    os.makedirs(DATA, exist_ok=True)
    fields = ["id", "granularity", "text", "valence", "arousal"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\n✓ 生成 {len(rows)} 篇 → {args.out}（去重共丟棄 {dropped_total} 篇）")

    if args.append:
        train_path = os.path.join(DATA, "train.csv")
        base_path = os.path.join(DATA, "train_base.csv")
        if not os.path.exists(base_path):  # 第一次先備份原始 train.csv
            with open(train_path, encoding="utf-8") as src, open(base_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            print(f"  已備份原始訓練集 → {base_path}")
        # 以 train_base.csv 為基底重組，避免重複 append
        with open(base_path, encoding="utf-8") as f:
            base_rows = list(csv.DictReader(f))
        with open(train_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(base_rows); w.writerows(rows)
        print(f"  ✓ train.csv = {len(base_rows)}（原始）+ {len(rows)}（增強）= {len(base_rows)+len(rows)} 筆")


if __name__ == "__main__":
    main()
