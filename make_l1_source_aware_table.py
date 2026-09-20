"""make_l1_source_aware_table.py — 把 2x2 ablation 的結果轉成論文用的 LaTeX 表。

對應 notebooks/Rocling2026_Colab_e22_l1_source_aware.ipynb（reviewer 1 指名補的
L1 x source-aware 2x2，含從未跑成的 E22 = L1 10 維 + source-aware）。

用法：
    # 還沒有結果時：產生 placeholder 表，論文仍可編譯（數字是紅色 TODO）
    python make_l1_source_aware_table.py

    # 有 dev proxy 結果後
    python make_l1_source_aware_table.py \
        --dev_summary ~/Downloads/l1_source_aware/dev_metrics_summary.csv \
        --dev_per_seed ~/Downloads/l1_source_aware/dev_metrics_per_seed.csv

    # 另外拿到官方分數後（official_scores_to_fill.csv 填好四個欄位）
    python make_l1_source_aware_table.py --dev_summary ... --official ...

產出：
    paper/tables/l1_source_aware_2x2.tex        dev proxy 2x2（mean ± sample SD）
    paper/tables/l1_source_aware_official.tex   官方分數表（--official 有填才產生）

論文目前的做法是**只在既有表格加一列 E22**（Table 4 設定索引、Table 5 official
validation），所以本檔預設只產生獨立的表格檔，並把「可直接貼進 Table 5 的那一列」
印出來。若之後要在論文放完整 2x2 表，先在 main.tex 補上
`% BEGIN l1-source-aware-2x2` / `% END l1-source-aware-2x2` 兩行標記，再加 --inject。
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(os.path.dirname(HERE), "paper")
TABLE_DIR = os.path.join(PAPER_DIR, "tables")
MAIN_TEX = os.path.join(PAPER_DIR, "main.tex")
BEGIN_MARK = "% BEGIN l1-source-aware-2x2"
END_MARK = "% END l1-source-aware-2x2"

# (condition key in the notebook, paper ID, lexicon label, source-aware label)
ROWS = [
    ("e4_l1", "E4", "L1 (10-d)", "No"),
    ("e22_l1_sa", "E22", "L1 (10-d)", "Yes"),
    ("e18_l1pp", "E18", "L1++ (31-d)", "No"),
    ("e19_l1pp_sa", "E19", "L1++ (31-d)", "Yes"),
]
METRICS = ["V_MAE", "V_PCC", "A_MAE", "A_PCC"]
TODO = r"\TODO{run}"


def read_csv_rows(path):
    if not path:
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt(value, digits=3):
    """0.8297 -> .830（與論文既有表格的省略前導零風格一致）。"""
    text = f"{float(value):.{digits}f}"
    return text[1:] if text.startswith("0.") else text


def cell(mean, sd):
    if mean is None or mean == "":
        return TODO
    if sd in (None, ""):
        return fmt(mean)
    return f"{fmt(mean)}$\\pm${fmt(sd)}"


def dev_table(summary_rows, n_seeds):
    by_condition = {r["condition"]: r for r in summary_rows if r.get("status") == "complete"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Lexicon $\times$ source-aware 2$\times$2 ablation on the 253-document"
        r" DSA-MST internal development proxy (mean $\pm$ sample SD over"
        f" {n_seeds} seeds)."
        r"  All rows use MacBERT, batch size 32, learning rate $2\times10^{-5}$, four"
        r" epochs, maximum length 256, and the same checkpoint criterion, so the"
        r" comparison is not confounded by the historical batch-size difference"
        r" between E4 and E18/E19.  E22 is the previously missing cell.}",
        r"\label{tab:l1-source-aware}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        r"ID & Lexicon & Src.\ wt. & V-MAE$\downarrow$ & V-PCC$\uparrow$"
        r" & A-MAE$\downarrow$ & A-PCC$\uparrow$ \\",
        r"\midrule",
    ]
    for key, paper_id, lex, src in ROWS:
        row = by_condition.get(key, {})
        values = [cell(row.get(f"{m}_mean"), row.get(f"{m}_sd")) for m in METRICS]
        lines.append(f"{paper_id} & {lex} & {src} & " + " & ".join(values) + r" \\")
        if key == "e22_l1_sa":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def official_table(official_rows):
    scored = [r for r in official_rows if any(r.get(m) for m in METRICS)]
    if not scored:
        return None
    by_key = {(r["condition"], str(r["seed"]), r["split"]): r for r in scored}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Official scores for the lexicon $\times$ source-aware 2$\times$2"
        r" ablation.  Rows without a score were not submitted.}",
        r"\label{tab:l1-source-aware-official}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        r"ID & Split & Seed & V-MAE$\downarrow$ & V-PCC$\uparrow$"
        r" & A-MAE$\downarrow$ & A-PCC$\uparrow$ \\",
        r"\midrule",
    ]
    for key, paper_id, _lex, _src in ROWS:
        for split in ("validation", "test"):
            for seed in sorted({r["seed"] for r in scored}, key=lambda s: (s != "42", s)):
                row = by_key.get((key, str(seed), split))
                if not row or not any(row.get(m) for m in METRICS):
                    continue
                values = [cell(row.get(m), None) for m in METRICS]
                lines.append(f"{paper_id} & {split} & {seed} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def inject(main_tex, table_body):
    """把表格寫回 main.tex 的 BEGIN/END 標記之間（標記本身保留）。"""
    with open(main_tex, encoding="utf-8") as f:
        text = f.read()
    start = text.find(BEGIN_MARK)
    end = text.find(END_MARK)
    if start < 0 or end < 0 or end < start:
        print(f"跳過注入：{main_tex} 沒有 {BEGIN_MARK} / {END_MARK} 標記")
        return False
    head_end = text.index("\n", start) + 1
    updated = text[:head_end] + table_body.rstrip("\n") + "\n" + text[end:]
    with open(main_tex, "w", encoding="utf-8") as f:
        f.write(updated)
    return updated != text


def paste_row(official_rows, condition="e22_l1_sa", seed="42", split="validation",
              row_label="E22: 10-d lexicon + source-aware"):
    """印出可直接取代論文 Table 5 裡 E22 那列 \\TODO 的一行 LaTeX。"""
    for r in official_rows:
        if (r.get("condition") == condition and str(r.get("seed")) == str(seed)
                and r.get("split") == split and all(r.get(m) for m in METRICS)):
            values = " & ".join(f"{float(r[m]):.3f}" for m in METRICS)
            print(f"\n貼進 main.tex 的 tab:validation（取代 E22 那列）：")
            print(f"{row_label} & {values} \\\\")
            return True
    print(f"\n{condition} seed {seed} 的 {split} 官方分數還沒填，"
          f"論文 Table 5 的 E22 列先維持 \\TODO")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_summary", default=None, help="notebook 產出的 dev_metrics_summary.csv")
    ap.add_argument("--dev_per_seed", default=None, help="dev_metrics_per_seed.csv（只用來數 seed）")
    ap.add_argument("--official", default=None, help="填好分數的 official_scores_to_fill.csv")
    ap.add_argument("--table_dir", default=TABLE_DIR)
    ap.add_argument("--main_tex", default=MAIN_TEX)
    ap.add_argument("--inject", action="store_true",
                    help="把完整 2x2 表寫回 main.tex 的 BEGIN/END 標記之間（標記要先存在）")
    args = ap.parse_args()

    summary_rows = read_csv_rows(args.dev_summary)
    per_seed = read_csv_rows(args.dev_per_seed)
    n_seeds = len({r["seed"] for r in per_seed}) if per_seed else 3

    os.makedirs(args.table_dir, exist_ok=True)
    dev_path = os.path.join(args.table_dir, "l1_source_aware_2x2.tex")
    body = dev_table(summary_rows, n_seeds)
    with open(dev_path, "w", encoding="utf-8") as f:
        f.write(body)
    complete = sum(1 for r in summary_rows if r.get("status") == "complete")
    print(f"wrote {dev_path}  ({complete}/4 conditions filled, {n_seeds} seeds)")

    if args.inject:
        changed = inject(args.main_tex, body)
        print(f"{'updated' if changed else 'unchanged'} {args.main_tex}")

    official_rows = read_csv_rows(args.official)
    paste_row(official_rows)
    official = official_table(official_rows)
    if official:
        official_path = os.path.join(args.table_dir, "l1_source_aware_official.tex")
        with open(official_path, "w", encoding="utf-8") as f:
            f.write(official)
        print(f"wrote {official_path}")
    elif args.official:
        print("--official 裡沒有任何已填分數，跳過官方分數表")


if __name__ == "__main__":
    main()
