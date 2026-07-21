"""
predict.py — 推論專用（不訓練）。

動機：`train_v2.py` 只在訓練結束後對 `data/val_unlabeled.csv` 推論一次，
沒有 inference-only 路徑。Shared task 結束後我們需要用既有 checkpoint 對
任意檔案（特別是 `DSANIDF_TestSet.csv`）重跑推論，做論文的分布分析。

用法：
    # 用最終提交模型對官方 test set 推論
    python predict.py --ckpt outputs/e19_source_aware_best.pt \
        --lex_mode l1_intensity --input ../DSANIDF_TestSet.csv \
        --run_name e19_source_aware --split test

    # 對 10 維 L1 的模型（實驗 4 / E10 的各 seed）
    python predict.py --ckpt outputs/macbert_s42_best.pt --lex_mode l1 \
        --input ../DSANIDF_TestSet.csv --run_name macbert_s42 --split test

產出：
    outputs/preds/{run_name}_{split}.csv   ID,valence_pred,arousal_pred,model_name,split
    outputs/{run_name}_{split}_submission.csv   ID,Valence,Arousal（官方格式）

⚠️ `--lex_mode` / `--model` / `--pooling` 必須與訓練該 checkpoint 時一致，
   否則 state_dict 形狀不合會直接報錯（這是刻意的，避免默默載錯）。
   對照表見 docs/experiments.md：
     實驗 4 / E10 / E11 / E13 → --lex_mode l1
     E18 / E19 / E20          → --lex_mode l1_intensity
     E21                      → --lex_mode l1_intensity --pooling mean_cls_dim_attention
"""
import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from train import VADataset, VARegressor, pick_device, denorm, LABEL_MIN, LABEL_MAX
from train_v2 import VARegressorDimAttn, predict, build_featurizer

HERE = os.path.dirname(os.path.abspath(__file__))


def read_rows(path):
    """讀 csv 並把欄名正規化成 id / text（官方檔用 ID,Text，內部檔用 id,text）。"""
    with open(path, encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        low = {(k or "").strip().lower(): v for k, v in r.items()}
        if "text" not in low:
            raise SystemExit(f"{path} 缺少 text/Text 欄位，實際欄位：{list(r.keys())}")
        rows.append({"id": low.get("id", ""), "text": low["text"] or ""})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="outputs/{run}_best.pt")
    ap.add_argument("--input", required=True, help="要推論的 csv（需含 ID/Text 欄）")
    ap.add_argument("--run_name", default=None, help="輸出檔名前綴，預設由 ckpt 推得")
    ap.add_argument("--split", default="test", help="輸出檔名後綴與 split 欄位值")
    ap.add_argument("--model", default="hfl/chinese-macbert-base")
    ap.add_argument("--lex_mode", choices=["none", "l1", "l1l2", "l1_intensity"], default="l1")
    ap.add_argument("--l2_pkl", default=os.path.join(HERE, "outputs", "l2_word_va.pkl"))
    ap.add_argument("--pooling", choices=["mean", "mean_cls_dim_attention"], default="mean")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    run = args.run_name or os.path.basename(args.ckpt).replace("_best.pt", "")
    preds_dir = os.path.join(args.out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    rows = read_rows(args.input)
    print(f"input: {args.input}  ({len(rows)} rows)")

    device = pick_device()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    featurizer, lex_dim = build_featurizer(args)

    model_cls = VARegressorDimAttn if args.pooling == "mean_cls_dim_attention" else VARegressor
    model = model_cls(args.model, lex_dim=lex_dim).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"loaded: {args.ckpt}  (lex_mode={args.lex_mode}, lex_dim={lex_dim}, pooling={args.pooling})")

    loader = DataLoader(
        VADataset(rows, tok, args.max_len, has_label=False, featurizer=featurizer),
        batch_size=args.batch_size)
    pred = np.clip(predict(model, loader, device), LABEL_MIN, LABEL_MAX)

    pred_path = os.path.join(preds_dir, f"{run}_{args.split}.csv")
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_pred", "arousal_pred", "model_name", "split"])
        for r, p in zip(rows, pred):
            w.writerow([r["id"], f"{p[0]:.4f}", f"{p[1]:.4f}", run, args.split])

    sub_path = os.path.join(args.out_dir, f"{run}_{args.split}_submission.csv")
    with open(sub_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Valence", "Arousal"])
        for r, (v, a) in zip(rows, pred):
            w.writerow([r["id"], f"{v:.4f}", f"{a:.4f}"])

    print(f"preds      -> {pred_path}")
    print(f"submission -> {sub_path}")
    print(f"[dist] valence mean={pred[:, 0].mean():.3f} std={pred[:, 0].std():.3f} | "
          f"arousal mean={pred[:, 1].mean():.3f} std={pred[:, 1].std():.3f}")


if __name__ == "__main__":
    main()
