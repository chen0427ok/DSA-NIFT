"""
Silver ranking benchmark（第一步）— 官方 200 篇無標籤文本的 LLM pairwise 排序標註。

背景：DSA-NIFT 是 zero in-domain labeled 任務，dev（DSA-MST 醫療反思）對增強實驗失真。
本腳本不產生絕對 VA 分數（LLM 校準不可靠，E13 已證明），而是對官方 `val_unlabeled.csv`
抽 pair、請 LLM 判斷「哪篇 arousal / valence 較高」，產出 silver pairwise 排序基準。
之後用 eval_silver_ranking.py 評各 run 的 val 預測與此排序的一致性（直攻 A_PCC 瓶頸）。

設計重點：
- Pair 抽樣只依 --seed 決定（與 judge 無關）→ 不同 judge 標的是同一批 pair，可做一致性過濾。
- 每個 pair 的 A/B 呈現順序由 hash(pair, tag) 決定 → 不同 judge 看到不同順序，抵銷位置偏差。
- 邊標邊寫檔（append）→ 中斷後重跑同指令自動續標。
- 多 judge：換 --model / --provider 各跑一次（--tag 區分），eval 時取一致 pair。

用法（在自己的終端機跑，Claude Code sandbox 沒有 API key；~500 次小呼叫，Opus 約 $2-3）：
    export ANTHROPIC_API_KEY=sk-ant-...
    python build_silver_pairs.py --n_pairs 500                     # judge 1: claude-opus-4-8
    export OPENAI_API_KEY=sk-...
    python build_silver_pairs.py --n_pairs 500 --provider openai   # judge 2: gpt-4o

    python build_silver_pairs.py --dry_run    # 不呼叫 API，只印 pair 統計與 prompt 範例

產出：data/silver_pairs_{tag}.csv
    pair_id,id_a,id_b,arousal_winner,valence_winner,judge,model
    winner 欄位存「勝出文本的 id」或 "tie"。
"""
import os
import csv
import random
import hashlib
import argparse

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
DEFAULT_MODELS = {"anthropic": "claude-opus-4-8", "openai": "gpt-4o"}

PROMPT_TMPL = """你是中文情緒維度分析（dimensional sentiment analysis）標註專家。以下兩段文本皆為新住民（移居台灣者）的第一人稱生活反思。

【定義】
- Arousal（喚醒度）：情緒的生理/心理激動、緊繃程度，與正負面無關。恐慌、狂喜、憤怒都是高喚醒；平靜、麻木、疲憊、淡然是低喚醒。
- Valence（效價）：情緒的正負面程度。喜悅、滿足是高效價；悲傷、憤怒、絕望是低效價。

【文本 A】
{text_a}

【文本 B】
{text_b}

【任務】分別判斷：
1. 哪段文本的 Arousal 較高？
2. 哪段文本的 Valence 較高？
兩者獨立判斷。只有在真的難分軒輊時才回 tie（請盡量少用 tie）。"""


def build_pairs(ids, n_pairs, seed):
    """只依 seed 決定的 pair 抽樣：多輪 shuffle 後相鄰配對，每篇出現次數平均。"""
    rng = random.Random(seed)
    pairs, seen = [], set()
    while len(pairs) < n_pairs:
        order = ids[:]
        rng.shuffle(order)
        for i in range(0, len(order) - 1, 2):
            key = frozenset((order[i], order[i + 1]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((order[i], order[i + 1]))
            if len(pairs) >= n_pairs:
                break
    return pairs


def present_order(id1, id2, tag):
    """由 hash(pair, tag) 決定 A/B 呈現順序：同 judge 可重現、不同 judge 順序不同。"""
    h = hashlib.md5(f"{id1}|{id2}|{tag}".encode()).hexdigest()
    return (id1, id2) if int(h, 16) % 2 == 0 else (id2, id1)


def make_client(provider):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()          # 讀 ANTHROPIC_API_KEY
    from openai import OpenAI
    return OpenAI()                            # 讀 OPENAI_API_KEY


def judge_pair(provider, client, model, ParsedModel, text_a, text_b):
    prompt = PROMPT_TMPL.format(text_a=text_a, text_b=text_b)
    if provider == "anthropic":
        resp = client.messages.parse(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=ParsedModel,
        )
        return resp.parsed_output
    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=ParsedModel,
    )
    return resp.choices[0].message.parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.join(DATA, "val_unlabeled.csv"))
    ap.add_argument("--n_pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42,
                    help="pair 抽樣 seed；多個 judge 必須用同一個 seed 才會標同一批 pair")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None, help="覆寫模型；預設依 provider")
    ap.add_argument("--tag", default=None, help="judge 標籤（預設 = 模型短名），決定輸出檔名")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    tag = args.tag or model.replace("/", "_").replace(".", "-")
    out_path = os.path.join(DATA, f"silver_pairs_{tag}.csv")

    with open(args.val, encoding="utf-8") as f:
        rows = {r["id"]: r["text"] for r in csv.DictReader(f)}
    ids = sorted(rows)
    pairs = build_pairs(ids, args.n_pairs, args.seed)
    per_text = args.n_pairs * 2 / len(ids)
    print(f"官方文本 {len(ids)} 篇 → {len(pairs)} pairs（每篇平均出現 {per_text:.1f} 次）")
    print(f"judge={tag} model={model} provider={args.provider} → {out_path}")

    if args.dry_run:
        a_id, b_id = present_order(*pairs[0], tag)
        print("\n--- 第一個 pair 的 prompt 範例 ---")
        print(PROMPT_TMPL.format(text_a=rows[a_id][:80] + "…", text_b=rows[b_id][:80] + "…"))
        return

    # 續標：已標過的 pair 直接跳過
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {frozenset((r["id_a"], r["id_b"])) for r in csv.DictReader(f)}
        print(f"已有 {len(done)} pairs，續標剩下的")

    from pydantic import BaseModel
    from typing import Literal

    class PairJudgment(BaseModel):
        arousal_higher: Literal["A", "B", "tie"]
        valence_higher: Literal["A", "B", "tie"]

    client = make_client(args.provider)
    fields = ["pair_id", "id_a", "id_b", "arousal_winner", "valence_winner", "judge", "model"]
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        n_done = len(done)
        for i, (id1, id2) in enumerate(pairs):
            if frozenset((id1, id2)) in done:
                continue
            a_id, b_id = present_order(id1, id2, tag)
            judgment = None
            for attempt in range(args.max_retries):
                try:
                    judgment = judge_pair(args.provider, client, model, PairJudgment,
                                          rows[a_id], rows[b_id])
                    break
                except Exception as e:
                    print(f"  pair {i} 失敗（{attempt+1}/{args.max_retries}）: {type(e).__name__}: {e}")
            if judgment is None:
                print(f"  ⚠ pair {i} 放棄（重跑同指令可續標）")
                continue

            def winner(choice):
                return {"A": a_id, "B": b_id, "tie": "tie"}[choice]

            w.writerow({"pair_id": f"P{i:04d}", "id_a": a_id, "id_b": b_id,
                        "arousal_winner": winner(judgment.arousal_higher),
                        "valence_winner": winner(judgment.valence_higher),
                        "judge": tag, "model": model})
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(pairs)} pairs 完成")
    print(f"✓ 完成 → {out_path}（{n_done} pairs）")


if __name__ == "__main__":
    main()
