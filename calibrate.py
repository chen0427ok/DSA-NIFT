"""
E17 — Arousal 後校準（post calibration）

對已有的預測做保守線性校準：
    pred' = m_dev + s * (pred - m_dev) + b     （m_dev = dev 預測平均，s/b 由 dev 網格搜尋）

⚠️ 重要：正線性變換 **不改變 PCC**，這一步只能修 MAE（arousal 預測整體偏移/壓縮的
校準問題），排序增益要靠模型本身。搜尋目標即 dev 上該維度的 MAE 最小。

用法（接在 ensemble.py 之後，對其統一預測檔做）：
    python calibrate.py --run e12_enc_ens --dim arousal --name e12_cal

產出：outputs/{name}_submission.csv（另一維原樣通過，全部 clip 到 [1,9]）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pred_dir/{run}_dev.csv 與 _val.csv")
    ap.add_argument("--pred_dir", default=os.path.join(HERE, "outputs", "preds"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--dim", choices=DIMS, default="arousal")
    ap.add_argument("--name", default=None)
    ap.add_argument("--scales", default="0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3")
    ap.add_argument("--shifts", default="-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4")
    args = ap.parse_args()
    name = args.name or f"{args.run}_cal"
    j = DIMS.index(args.dim)

    _, dev_pred, gold = read_pred(os.path.join(args.pred_dir, f"{args.run}_dev.csv"), True)
    val_ids, val_pred, _ = read_pred(os.path.join(args.pred_dir, f"{args.run}_val.csv"), False)

    p, g = dev_pred[:, j], gold[:, j]
    m = p.mean()
    base_mae = np.mean(np.abs(p - g))
    pcc = float(np.corrcoef(p, g)[0, 1])
    print(f"[{args.run}] dev {args.dim}: MAE={base_mae:.4f} PCC={pcc:.4f} "
          f"pred mean/std={m:.3f}/{p.std():.3f} gold mean/std={g.mean():.3f}/{g.std():.3f}")
    print("（提醒：線性校準不改 PCC，只修 MAE）\n")

    best = (base_mae, 1.0, 0.0)
    for s in (float(x) for x in args.scales.split(",")):
        for b in (float(x) for x in args.shifts.split(",")):
            cal = np.clip(m + s * (p - m) + b, 1.0, 9.0)
            mae = np.mean(np.abs(cal - g))
            if mae < best[0] - 1e-9:
                best = (mae, s, b)
    mae, s, b = best
    print(f"最佳: scale={s} shift={b}  dev {args.dim} MAE {base_mae:.4f} -> {mae:.4f} "
          f"(Δ={mae-base_mae:+.4f})")
    if s == 1.0 and b == 0.0:
        print("dev 上無利可圖，輸出的 submission 與未校準相同。")

    out = val_pred.copy()
    out[:, j] = m + s * (out[:, j] - m) + b   # 用 dev 的錨點，同一組變換套到 val
    out = np.clip(out, 1.0, 9.0)
    sub = os.path.join(args.out_dir, f"{name}_submission.csv")
    with open(sub, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Valence", "Arousal"])
        for ID, (v, a) in zip(val_ids, out):
            w.writerow([ID, f"{v:.4f}", f"{a:.4f}"])
    print(f"submission -> {sub}")


if __name__ == "__main__":
    main()
