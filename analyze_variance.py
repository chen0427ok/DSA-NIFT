"""
analyze_variance.py — 量化「在 n=200 的驗證集上，PCC 估計有多不穩」。

動機（docs/experiments.md §5.4）：我們用 13 次提交在一個 200 篇的 leaderboard 上挑最佳模型，
選出的 E19 在 validation 有 A_PCC 0.452，到 1,100 篇 test 只剩 0.357，而 valence 幾乎不動。
本腳本用「有 gold 標籤的內部 dev（253 篇）」做 bootstrap，估計 n=200 時
A_PCC 與 V_PCC 的抽樣分布寬度，回答：**我們據以做決策的那些差距，是否根本落在雜訊裡。**

用法：
    .venv/bin/python analyze_variance.py                 # 全部 run
    .venv/bin/python analyze_variance.py --n 200 --iters 5000

產出：
    stdout 表格 + outputs/analysis/dev_bootstrap_ci.csv
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def pcc(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_run(df, n, iters, rng):
    """對單一 run 的 dev 預測做 bootstrap：每次抽 n 篇（可重複），算四個指標。"""
    N = len(df)
    vt, at = df.valence_true.values, df.arousal_true.values
    vp, ap = df.valence_pred.values, df.arousal_pred.values
    out = {k: np.empty(iters) for k in ["V_MAE", "V_PCC", "A_MAE", "A_PCC"]}
    for i in range(iters):
        idx = rng.integers(0, N, size=n)
        out["V_MAE"][i] = np.mean(np.abs(vp[idx] - vt[idx]))
        out["A_MAE"][i] = np.mean(np.abs(ap[idx] - at[idx]))
        out["V_PCC"][i] = pcc(vp[idx], vt[idx])
        out["A_PCC"][i] = pcc(ap[idx], at[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="每次 bootstrap 抽樣數，預設 200 = 官方 validation 規模")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preds_dir", default=os.path.join(HERE, "outputs", "preds"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "outputs", "analysis"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.preds_dir, "*_dev.csv")))
    if not files:
        raise SystemExit(f"找不到 dev 預測檔於 {args.preds_dir}")

    rows, point = [], {}
    for f in files:
        run = os.path.basename(f)[:-8]
        df = pd.read_csv(f)
        bs = bootstrap_run(df, args.n, args.iters, rng)
        rec = {"run": run, "n_dev": len(df)}
        for k in ["V_MAE", "V_PCC", "A_MAE", "A_PCC"]:
            v = bs[k][~np.isnan(bs[k])]
            lo, hi = np.percentile(v, [2.5, 97.5])
            rec[f"{k}_point"] = {
                "V_MAE": np.mean(np.abs(df.valence_pred - df.valence_true)),
                "A_MAE": np.mean(np.abs(df.arousal_pred - df.arousal_true)),
                "V_PCC": pcc(df.valence_pred.values, df.valence_true.values),
                "A_PCC": pcc(df.arousal_pred.values, df.arousal_true.values),
            }[k]
            rec[f"{k}_lo"], rec[f"{k}_hi"] = lo, hi
            rec[f"{k}_ci_width"] = hi - lo
        rows.append(rec)
        point[run] = rec

    res = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "dev_bootstrap_ci.csv")
    res.to_csv(out_csv, index=False)

    print(f"\n=== Bootstrap (n={args.n}, iters={args.iters}) on internal dev ===\n")
    show = res[["run", "V_PCC_point", "V_PCC_ci_width", "A_PCC_point", "A_PCC_ci_width",
                "V_MAE_point", "V_MAE_ci_width", "A_MAE_point", "A_MAE_ci_width"]]
    print(show.round(4).to_string(index=False))

    print("\n=== 核心對比：n=200 時 PCC 的 95% CI 寬度 ===")
    print(f"  V_PCC 平均 CI 寬度: {res.V_PCC_ci_width.mean():.4f}")
    print(f"  A_PCC 平均 CI 寬度: {res.A_PCC_ci_width.mean():.4f}")
    ratio = res.A_PCC_ci_width.mean() / max(res.V_PCC_ci_width.mean(), 1e-9)
    print(f"  → arousal 的評估不確定性是 valence 的 {ratio:.2f} 倍")

    print("\n=== 我們在 official validation 上據以決策的差距 ===")
    print("  E4 (0.426) vs E19 (0.452) 的 A_PCC 差距 = 0.026")
    print(f"  n=200 的 A_PCC 95% CI 寬度 ≈ {res.A_PCC_ci_width.mean():.3f}")
    if res.A_PCC_ci_width.mean() > 0.026:
        print("  ⇒ 該差距【小於】抽樣噪聲寬度：validation 上的排名不可信（支持 selection overfitting 論點）")
    else:
        print("  ⇒ 該差距大於抽樣噪聲寬度：需要另尋 test 崩落的解釋")

    # seed 變異（僅純 seed 差異的 run）
    seeds = [r for r in res.run if r.startswith("macbert_s") and "pseudo" not in r]
    if len(seeds) > 1:
        sub = res[res.run.isin(seeds)]
        print(f"\n=== Seed 變異（{len(seeds)} 顆同設定不同 seed，dev 全量點估計）===")
        for k in ["V_PCC", "A_PCC"]:
            vals = sub[f"{k}_point"]
            print(f"  {k}: mean={vals.mean():.4f} std={vals.std():.4f} "
                  f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    print(f"\n儲存 -> {out_csv}")


if __name__ == "__main__":
    main()
