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

from affective_graph import build_graph, seeds_for_target

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")
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


def build_prompt(seed_words, desc, n):
    seeds = "、".join(seed_words) if seed_words else "（圖上無此區間種子詞，請自行掌握情緒強度）"
    return (
        f"你要協助建立中文情緒分析訓練資料。請以「新住民（嫁來/移居台灣的外籍配偶或移工）」的第一人稱，"
        f"寫 {n} 段彼此獨立、內容各異的生活反思短文。\n\n"
        f"【情緒要求】每段都要明確傳達：{desc}。情緒強度要到位、可從文字明顯感受到。\n"
        f"【可用情緒詞】可自然帶入這些詞當情緒參考（不必全用、不要硬塞；"
        f"凡與新住民日常語境不符、過於暴力或粗俗的詞請直接忽略）：{seeds}\n"
        f"【內容要求】題材多元：工作、家庭、語言、孩子教育、思鄉、人際、就醫、證件/身分、節慶等都可；"
        f"每段 50–120 字；用第一人稱「我」；像真人日記/自述，不要像新聞或教科書；不要編號、不要標題。\n"
        f"只回傳這 {n} 段短文。"
    )


def make_client(provider):
    """建立 LLM client（讀對應的環境變數 API key）。"""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()          # 讀 ANTHROPIC_API_KEY
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI()                        # 讀 OPENAI_API_KEY
    raise ValueError(f"未知 provider: {provider}")


def generate_bin(provider, client, model, ParsedModel, seed_words, desc, n):
    """呼叫 LLM 生成 n 段短文，回傳 list[str]。兩家都用 structured output 拿乾淨陣列。"""
    prompt = build_prompt(seed_words, desc, n)
    if provider == "anthropic":
        resp = client.messages.parse(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
            output_format=ParsedModel,
        )
        texts = resp.parsed_output.texts
    else:  # openai
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ParsedModel,
        )
        texts = resp.choices[0].message.parsed.texts
    return [t.strip() for t in texts if t and t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_bin", type=int, default=40, help="每個 VA 區間要生幾篇")
    ap.add_argument("--chunk", type=int, default=20, help="每次 API 呼叫請幾篇（湊滿 per_bin 會多次呼叫）")
    ap.add_argument("--jitter", type=float, default=0.4, help="VA 標籤在中心值附近的隨機抖動幅度")
    ap.add_argument("--seeds_per_bin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None, help="覆寫模型；預設依 provider（claude-opus-4-8 / gpt-4o）")
    ap.add_argument("--out", default=os.path.join(DATA, "train_aug.csv"))
    ap.add_argument("--append", action="store_true", help="生成後併進 data/train.csv（先備份 train_base.csv）")
    ap.add_argument("--dry_run", action="store_true", help="不呼叫 API，只印種子詞與 prompt")
    args = ap.parse_args()
    random.seed(args.seed)
    model = args.model or DEFAULT_MODELS[args.provider]

    G = load_graph()

    client = ParsedModel = None
    if not args.dry_run:
        from pydantic import BaseModel

        class GeneratedTexts(BaseModel):
            texts: list[str]

        ParsedModel = GeneratedTexts
        client = make_client(args.provider)
        print(f"provider={args.provider}  model={model}")

    rows = []
    for i, (vc, ac, vr, ar, desc) in enumerate(TARGET_BINS):
        seeds = seeds_for_target(G, v_range=vr, a_range=ar, n=args.seeds_per_bin)
        seed_words = [w for w, _, _ in seeds]
        print(f"\n[bin {i+1}/{len(TARGET_BINS)}] V≈{vc} A≈{ac}  {desc}")
        print(f"  種子詞({len(seed_words)}): {'、'.join(seed_words) or '（無）'}")

        if args.dry_run:
            print("  --- prompt 預覽 ---")
            print("  " + build_prompt(seed_words, desc, min(args.chunk, args.per_bin)).replace("\n", "\n  "))
            continue

        texts = []
        while len(texts) < args.per_bin:
            need = min(args.chunk, args.per_bin - len(texts))
            got = generate_bin(args.provider, client, model, ParsedModel, seed_words, desc, need)
            if not got:
                print("  ⚠ 這批回傳 0 篇，跳出避免無限迴圈"); break
            texts.extend(got)
            print(f"  已生成 {len(texts)}/{args.per_bin}")

        for t in texts[:args.per_bin]:
            v = round(min(9.0, max(1.0, vc + random.uniform(-args.jitter, args.jitter))), 2)
            a = round(min(9.0, max(1.0, ac + random.uniform(-args.jitter, args.jitter))), 2)
            rows.append({"id": f"AUG_{len(rows)+1:04d}", "granularity": "augment",
                         "text": t, "valence": v, "arousal": a})

    if args.dry_run:
        print(f"\n（dry_run）共 {len(TARGET_BINS)} 個 VA 區間，未呼叫 API、未輸出檔案。")
        return

    os.makedirs(DATA, exist_ok=True)
    fields = ["id", "granularity", "text", "valence", "arousal"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\n✓ 生成 {len(rows)} 篇 → {args.out}")

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
