"""
Silver ranking benchmark（第二步）— 用 LLM pairwise 排序評各 run 的官方 val 預測。

讀 build_silver_pairs.py 產生的 data/silver_pairs_*.csv（可多個 judge），
與 outputs/preds/{run}_val.csv（train_v2.py 的統一預測輸出），計算：

    pairwise accuracy = 模型預測方向與 silver 排序一致的 pair 比例
    （預測相等算 0.5；等價於 concordance ≈ (Kendall τ + 1) / 2）

多個 judge 時只取「所有 judge 一致且非 tie」的 pair（一致性過濾，抵銷單一 judge
的偏誤與位置偏差）。這個指標是目標域的排序訊號，直接對應 A_PCC 瓶頸——
用它決定哪個 run 值得花官方提交額度，不要只看 DSA-MST dev。

用法（本機跑，不需 API / GPU）：
    python eval_silver_ranking.py                          # 評 outputs/preds/ 下所有 run
    python eval_silver_ranking.py --runs e18_l1_intensity e20_rank_aug
    python eval_silver_ranking.py --allow_majority         # 多 judge 時放寬為多數決
"""
import os
import csv
import glob
import argparse
from collections import defaultdict

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
PREDS = os.path.join(HERE, "outputs", "preds")


def load_silver(paths, allow_majority=False):
    """彙整多 judge → {dim: {frozenset(id1,id2): winner_id}}，只保留可信 pair。"""
    votes = {"arousal": defaultdict(list), "valence": defaultdict(list)}
    judges = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                judges.add(r["judge"])
                key = frozenset((r["id_a"], r["id_b"]))
                votes["arousal"][key].append(r["arousal_winner"])
                votes["valence"][key].append(r["valence_winner"])

    silver = {}
    for dim, pair_votes in votes.items():
        kept = {}
        for key, ws in pair_votes.items():
            non_tie = [w for w in ws if w != "tie"]
            if not non_tie:
                continue
            if allow_majority:
                top = max(set(non_tie), key=non_tie.count)
                if non_tie.count(top) * 2 > len(ws):  # 過半（tie 算反對票）
                    kept[key] = top
            else:
                if len(set(ws)) == 1 and ws[0] != "tie":  # 全體一致且非 tie
                    kept[key] = ws[0]
        silver[dim] = kept
    return silver, sorted(judges)


def load_val_preds(path):
    with open(path, encoding="utf-8") as f:
        return {r["ID"]: (float(r["valence_pred"]), float(r["arousal_pred"]))
                for r in csv.DictReader(f)}


def pairwise_acc(preds, silver_dim, col):
    """col: 0=valence 1=arousal。回傳 (accuracy, n_pairs)。預測相等算 0.5。"""
    score, n = 0.0, 0
    for key, winner in silver_dim.items():
        loser = next(i for i in key if i != winner)
        if winner not in preds or loser not in preds:
            continue
        d = preds[winner][col] - preds[loser][col]
        score += 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        n += 1
    return (score / n if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="silver pairs csv（預設 data/silver_pairs_*.csv 全部）")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="要評的 run 名（預設 outputs/preds/ 下所有 *_val.csv）")
    ap.add_argument("--preds_dir", default=PREDS)
    ap.add_argument("--allow_majority", action="store_true",
                    help="多 judge 時用多數決（預設要求全體一致）")
    args = ap.parse_args()

    pair_files = args.pairs or sorted(glob.glob(os.path.join(DATA, "silver_pairs_*.csv")))
    if not pair_files:
        raise SystemExit("找不到 silver pairs（先跑 build_silver_pairs.py）")
    silver, judges = load_silver(pair_files, args.allow_majority)
    mode = "多數決" if args.allow_majority else "全體一致"
    print(f"judges: {', '.join(judges)}（{mode}過濾）")
    print(f"可信 pairs: arousal={len(silver['arousal'])}  valence={len(silver['valence'])}\n")

    if args.runs:
        val_files = [os.path.join(args.preds_dir, f"{r}_val.csv") for r in args.runs]
    else:
        val_files = sorted(glob.glob(os.path.join(args.preds_dir, "*_val.csv")))
    if not val_files:
        raise SystemExit(f"找不到 val 預測檔（{args.preds_dir}/*_val.csv）")

    results = []
    for p in val_files:
        run = os.path.basename(p)[:-8]
        preds = load_val_preds(p)
        a_acc, a_n = pairwise_acc(preds, silver["arousal"], col=1)
        v_acc, v_n = pairwise_acc(preds, silver["valence"], col=0)
        results.append((run, a_acc, a_n, v_acc, v_n))

    results.sort(key=lambda r: -(r[1] if r[1] == r[1] else -1))  # 依 arousal acc 排序
    print(f"{'run':36s} {'A_rank_acc':>10s} {'(pairs)':>8s} {'V_rank_acc':>10s} {'(pairs)':>8s}")
    for run, a_acc, a_n, v_acc, v_n in results:
        print(f"{run:36s} {a_acc:10.4f} {a_n:8d} {v_acc:10.4f} {v_n:8d}")
    print("\n0.5 = 隨機；越高 = 目標域排序越接近 silver 基準（arousal 為主要決策依據）")


if __name__ == "__main__":
    main()
