"""
E10 / E12 — Dimension-wise Weighted Ensemble

讀多個 run 的統一預測檔（train_v2.py / embed_regressor.py 產出的
outputs/preds/{run}_dev.csv 與 {run}_val.csv），在 dev 上算每個模型的 MAE/PCC，
valence / arousal **分開**算權重後融合 val 預測，輸出官方格式 submission。

權重（--mode weighted，pipeline 文件 3.1）：
    score_m,d = max(PCC_m,d, 0) / (MAE_m,d + 1e-6)
    w_m,d     = score_m,d / Σ_m score_m,d
--mode mean 則等權平均（multi-seed ensemble 用這個即可）。

用法：
    # E10：同 encoder 5 seeds 等權平均
    python ensemble.py macbert_s42 macbert_s1 macbert_s2 macbert_s3 macbert_s4 \
        --mode mean --name e10_seed_ens
    # E12：跨 encoder，dimension-wise 加權
    python ensemble.py macbert_s42 roberta_s42 robertaL_s42 svr_macbert \
        --mode weighted --name e12_enc_ens

產出：
    outputs/{name}_submission.csv          官方格式
    outputs/preds/{name}_dev.csv / _val.csv 統一格式（可再餵 calibrate.py 或當另一個 run 疊 ensemble）
"""
import os
import csv
import argparse
import numpy as np

HERE = os.path.dirname(__file__)
DIMS = ["valence", "arousal"]


def read_pred(path, has_true):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = [r["ID"] for r in rows]
    pred = np.array([[float(r["valence_pred"]), float(r["arousal_pred"])] for r in rows])
    gold = None
    if has_true:
        gold = np.array([[float(r["valence_true"]), float(r["arousal_true"])] for r in rows])
    return ids, pred, gold


def metrics(pred, gold):
    out = {}
    for j, d in enumerate(DIMS):
        out[f"{d}_MAE"] = float(np.mean(np.abs(pred[:, j] - gold[:, j])))
        out[f"{d}_PCC"] = float(np.corrcoef(pred[:, j], gold[:, j])[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 名稱（對應 pred_dir/{run}_dev.csv 與 _val.csv）")
    ap.add_argument("--pred_dir", default=os.path.join(HERE, "outputs", "preds"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--mode", choices=["weighted", "mean"], default="weighted")
    ap.add_argument("--name", default="ensemble")
    args = ap.parse_args()

    dev_preds, val_preds, per_model = {}, {}, {}
    ref_dev_ids = ref_val_ids = None
    gold = None
    for run in args.runs:
        d_ids, d_pred, d_gold = read_pred(os.path.join(args.pred_dir, f"{run}_dev.csv"), True)
        v_ids, v_pred, _ = read_pred(os.path.join(args.pred_dir, f"{run}_val.csv"), False)
        if ref_dev_ids is None:
            ref_dev_ids, ref_val_ids, gold = d_ids, v_ids, d_gold
        assert d_ids == ref_dev_ids and v_ids == ref_val_ids, f"{run} 的 ID 順序與其他 run 不一致"
        dev_preds[run], val_preds[run] = d_pred, v_pred
        per_model[run] = metrics(d_pred, gold)

    # dimension-wise 權重
    W = np.zeros((len(args.runs), 2))
    for i, run in enumerate(args.runs):
        for j, d in enumerate(DIMS):
            m = per_model[run]
            W[i, j] = 1.0 if args.mode == "mean" else max(m[f"{d}_PCC"], 0.0) / (m[f"{d}_MAE"] + 1e-6)
    W = W / W.sum(axis=0, keepdims=True)

    print(f"=== 各模型 dev 表現與權重 ({args.mode}) ===")
    for i, run in enumerate(args.runs):
        m = per_model[run]
        print(f"{run:24s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
              + f"  | w_V={W[i,0]:.3f} w_A={W[i,1]:.3f}")

    dev_ens = sum(W[i] * dev_preds[run] for i, run in enumerate(args.runs))
    val_ens = sum(W[i] * val_preds[run] for i, run in enumerate(args.runs))
    val_ens = np.clip(val_ens, 1.0, 9.0)
    print("=== Ensemble dev ===")
    print("  " + " ".join(f"{k}={v:.4f}" for k, v in metrics(dev_ens, gold).items()))

    os.makedirs(args.pred_dir, exist_ok=True)
    with open(os.path.join(args.pred_dir, f"{args.name}_dev.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_true", "arousal_true", "valence_pred", "arousal_pred", "model_name", "split"])
        for i, ID in enumerate(ref_dev_ids):
            w.writerow([ID, f"{gold[i,0]:.4f}", f"{gold[i,1]:.4f}",
                        f"{dev_ens[i,0]:.4f}", f"{dev_ens[i,1]:.4f}", args.name, "dev"])
    with open(os.path.join(args.pred_dir, f"{args.name}_val.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "valence_pred", "arousal_pred", "model_name", "split"])
        for i, ID in enumerate(ref_val_ids):
            w.writerow([ID, f"{val_ens[i,0]:.4f}", f"{val_ens[i,1]:.4f}", args.name, "val"])

    sub = os.path.join(args.out_dir, f"{args.name}_submission.csv")
    with open(sub, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Valence", "Arousal"])
        for i, ID in enumerate(ref_val_ids):
            w.writerow([ID, f"{val_ens[i,0]:.4f}", f"{val_ens[i,1]:.4f}"])
    print(f"submission -> {sub}")


if __name__ == "__main__":
    main()
