"""Generate the source-aware sensitivity Colab notebook deterministically."""
import json
import pathlib
import textwrap


HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE / "Rocling2026_Colab_source_aware_sensitivity.ipynb"


def cell(kind, source):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


def build_notebook():
    cells = [
        cell("markdown", r'''
        # ROCLING 2026：Source-aware weight sensitivity（4 configs × 3 seeds）

        這是固定矩陣的 sensitivity check，不是 hyperparameter search。依序執行全部 cells；
        中斷後可重跑訓練 cell，已完整保存到 Drive 的 run 會通過 receipt 驗證後跳過。

        | config | CVAS | CVAT | DSA-MST | EDU2021 |
        |---|---:|---:|---:|---:|
        | uniform | 1.00 | 1.00 | 1.00 | 1.00 |
        | mild | 0.50 | 0.75 | 1.00 | 1.00 |
        | current | 0.25 | 0.50 | 1.00 | 0.75 |
        | reflection_swap | 0.25 | 0.50 | 0.75 | 1.00 |
        '''),
        cell("code", r'''
        # 1) 安裝依賴並確認 GPU
        %pip install -q "transformers>=4.40,<5" "pandas>=2,<3" "numpy>=1.24,<3" jieba
        import torch
        assert torch.cuda.is_available(), "請在 Runtime > Change runtime type 選 GPU 後重跑"
        print(torch.cuda.get_device_name(0))
        '''),
        cell("code", r'''
        # 2) 掛載 Drive 並取得 repo（private repo token 不寫入 notebook）
        from google.colab import drive
        drive.mount('/content/drive')
        import getpass, os, pathlib, subprocess

        REPO_URL = "https://github.com/chen0427ok/DSA-NIFT.git"
        REPO_PATH = pathlib.Path("/content/DSA-NIFT")
        RESULT_ROOT = pathlib.Path("/content/drive/MyDrive/ROCLING2026_source_aware_sensitivity")
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        if not (REPO_PATH / ".git").exists():
            token = getpass.getpass("GitHub token（repo 公開則直接 Enter）: ").strip()
            clone_url = REPO_URL if not token else REPO_URL.replace("https://", f"https://x-access-token:{token}@")
            subprocess.run(["git", "clone", "--branch", "main", clone_url, str(REPO_PATH)], check=True)
            del token, clone_url
        os.chdir(REPO_PATH)
        REPO_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        print("repo", REPO_PATH, REPO_COMMIT)
        '''),
        cell("code", r'''
        # 3) 實驗矩陣、pure helpers 與預檢
        # PURE_HELPERS
        import csv
        import hashlib
        import json
        import os
        import pathlib
        import shutil
        import subprocess
        import sys
        import time
        import numpy as np
        import pandas as pd

        SEEDS = [42, 1, 2]
        CONFIGS = {
            "uniform": {"sentence": 1.00, "text": 1.00, "reflection": 1.00, "edu2021": 1.00},
            "mild": {"sentence": 0.50, "text": 0.75, "reflection": 1.00, "edu2021": 1.00},
            "current": {"sentence": 0.25, "text": 0.50, "reflection": 1.00, "edu2021": 0.75},
            "reflection_swap": {"sentence": 0.25, "text": 0.50, "reflection": 0.75, "edu2021": 1.00},
        }
        MODEL_ID = "hfl/chinese-macbert-base"
        METRICS = ["V_MAE", "V_PCC", "A_MAE", "A_PCC"]
        REPO_PATH = pathlib.Path(globals().get("REPO_PATH", "."))
        RESULT_ROOT = pathlib.Path(globals().get("RESULT_ROOT", "outputs/source_aware_sensitivity"))

        def expected_runs():
            return [(config, seed) for config in CONFIGS for seed in SEEDS]

        def run_name(config, seed):
            return f"sa_sensitivity_{config}_s{seed}"

        def _fmt_weight(value):
            return f"{value:g}"

        def source_weight_spec(config):
            weights = CONFIGS[config]
            return ",".join(f"{src}=1:{_fmt_weight(weights[src])}"
                            for src in ("sentence", "text", "reflection", "edu2021"))

        def build_train_command(config, seed):
            return [sys.executable, "train_v2.py", "--model", MODEL_ID,
                    "--lex_mode", "l1_intensity", "--source_aware",
                    "--source_weights", source_weight_spec(config),
                    "--run_name", run_name(config, seed), "--seed", str(seed),
                    "--epochs", "4", "--batch_size", "32", "--lr", "2e-05",
                    "--max_len", "256"]

        def sha256(path):
            digest = hashlib.sha256()
            with pathlib.Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def compute_dev_metrics(prediction_path):
            frame = pd.read_csv(prediction_path)
            out = {}
            for prefix, gold, pred in (("V", "valence_true", "valence_pred"),
                                       ("A", "arousal_true", "arousal_pred")):
                y, yhat = frame[gold].to_numpy(float), frame[pred].to_numpy(float)
                out[f"{prefix}_MAE"] = float(np.mean(np.abs(y - yhat)))
                out[f"{prefix}_PCC"] = float(np.corrcoef(y, yhat)[0, 1]) if np.std(yhat) > 1e-12 else 0.0
            return out

        def aggregate_rows(rows):
            frame = pd.DataFrame(rows)
            actual = set(zip(frame.get("config", []), frame.get("seed", [])))
            required = set(expected_runs())
            if actual != required or len(frame) != 12:
                missing = sorted(required - actual)
                raise RuntimeError(f"incomplete 12-run matrix; missing={missing}")
            order = {name: i for i, name in enumerate(CONFIGS)}
            frame = frame.sort_values(["config", "seed"], key=lambda s: s.map(order) if s.name == "config" else s)
            summary_rows = []
            for config in CONFIGS:
                group = frame[frame.config == config]
                row = {"config": config}
                for metric in METRICS:
                    row[f"{metric}_mean"] = group[metric].mean()
                    row[f"{metric}_std"] = group[metric].std(ddof=1)
                summary_rows.append(row)
            base = frame[frame.config == "uniform"].set_index("seed")
            paired_rows = []
            for config in list(CONFIGS)[1:]:
                for _, row in frame[frame.config == config].iterrows():
                    paired = {"config": config, "seed": int(row.seed)}
                    for metric in METRICS:
                        paired[f"delta_{metric}_vs_uniform"] = row[metric] - base.loc[row.seed, metric]
                    paired_rows.append(paired)
            return frame.reset_index(drop=True), pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)

        def validate_inputs(repo_path):
            required = [repo_path / "train_v2.py", repo_path / "data/train.csv",
                        repo_path / "data/dev.csv", repo_path / "external/emobank/CVAW_all_SD.csv",
                        repo_path / "external/emobank/CVAP_all_SD.csv"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError("missing required files: " + ", ".join(missing))
            counts = {}
            for name in ("train", "dev"):
                path = repo_path / f"data/{name}.csv"
                frame = pd.read_csv(path)
                need = {"id", "granularity", "text", "valence", "arousal"}
                if not need.issubset(frame.columns):
                    raise ValueError(f"{path} missing columns {sorted(need - set(frame.columns))}")
                counts[name] = len(frame)
            found = set(pd.read_csv(repo_path / "data/train.csv").granularity)
            if not set(("sentence", "text", "reflection", "edu2021")).issubset(found):
                raise ValueError(f"train.csv granularities incomplete: {sorted(found)}")
            return counts

        print(validate_inputs(REPO_PATH))
        display(pd.DataFrame([{"config": c, "seed": s, "weights": source_weight_spec(c)}
                              for c, s in expected_runs()]))
        '''),
        cell("code", r'''
        # 4) 可續跑的訓練 helpers
        def atomic_write_json(path, payload):
            path = pathlib.Path(path)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)

        def valid_receipt(run_dir, config, seed):
            receipt_path = run_dir / "receipt.json"
            if not receipt_path.exists():
                return False
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["config"] != config or receipt["seed"] != seed:
                    return False
                return all((run_dir / item["name"]).exists() and
                           sha256(run_dir / item["name"]) == item["sha256"]
                           for item in receipt["artifacts"])
            except (KeyError, ValueError, OSError):
                return False

        def run_streamed(command, log_path):
            with pathlib.Path(log_path).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=REPO_PATH, text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           bufsize=1)
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)

        def run_one(config, seed):
            name = run_name(config, seed)
            run_dir = RESULT_ROOT / "runs" / name
            if valid_receipt(run_dir, config, seed):
                print("SKIP verified", name)
                return json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            run_streamed(build_train_command(config, seed), log_path)
            sources = {
                "checkpoint.pt": REPO_PATH / "outputs" / f"{name}_best.pt",
                "dev_predictions.csv": REPO_PATH / "outputs/preds" / f"{name}_dev.csv",
                "val_predictions.csv": REPO_PATH / "outputs/preds" / f"{name}_val.csv",
                "validation_submission.csv": REPO_PATH / "outputs" / f"{name}_submission.csv",
            }
            for target, source in sources.items():
                if not source.exists():
                    raise FileNotFoundError(f"training did not create {source}")
                shutil.copy2(source, run_dir / target)
            metrics = compute_dev_metrics(run_dir / "dev_predictions.csv")
            artifacts = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(run_dir.iterdir()) if p.is_file() and p.name != "receipt.json"]
            receipt = {"run_name": name, "config": config, "seed": seed,
                       "weights": CONFIGS[config], "metrics": metrics,
                       "command": build_train_command(config, seed),
                       "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "artifacts": artifacts}
            atomic_write_json(run_dir / "receipt.json", receipt)
            return receipt
        '''),
        cell("code", r'''
        # 5) 主迴圈：12 runs；失敗即停止，修正後重跑本 cell 可續跑
        statuses = []
        for config, seed in expected_runs():
            try:
                receipt = run_one(config, seed)
                statuses.append({"config": config, "seed": seed, "status": "complete", **receipt["metrics"]})
            except Exception as exc:
                statuses.append({"config": config, "seed": seed, "status": "FAILED", "error": repr(exc)})
                display(pd.DataFrame(statuses))
                raise
        display(pd.DataFrame(statuses))
        '''),
        cell("code", r'''
        # 6) 嚴格彙整、LaTeX、manifest 與完整 ZIP
        receipts = []
        for config, seed in expected_runs():
            path = RESULT_ROOT / "runs" / run_name(config, seed) / "receipt.json"
            if not valid_receipt(path.parent, config, seed):
                raise RuntimeError(f"incomplete 12-run matrix; invalid receipt: {config}/{seed}")
            receipts.append(json.loads(path.read_text(encoding="utf-8")))
        rows = [{"config": r["config"], "seed": r["seed"], **r["metrics"]} for r in receipts]
        per_run, summary, paired = aggregate_rows(rows)
        per_run.to_csv(RESULT_ROOT / "source_aware_per_run.csv", index=False)
        summary.to_csv(RESULT_ROOT / "source_aware_summary.csv", index=False)
        paired.to_csv(RESULT_ROOT / "source_aware_paired_vs_uniform.csv", index=False)
        latex = summary.to_latex(index=False, float_format=lambda x: f"{x:.4f}")
        (RESULT_ROOT / "source_aware_summary.tex").write_text(latex, encoding="utf-8")
        manifest = {"purpose": "fixed source-aware sensitivity analysis; not hyperparameter search",
                    "repo_commit": REPO_COMMIT, "model": MODEL_ID, "configs": CONFIGS,
                    "seeds": SEEDS, "runs": receipts, "python": sys.version,
                    "torch": torch.__version__}
        atomic_write_json(RESULT_ROOT / "experiment_manifest.json", manifest)
        display(per_run); display(summary); display(paired)

        archive_base = pathlib.Path("/content/source_aware_sensitivity_results")
        archive = pathlib.Path(shutil.make_archive(str(archive_base), "zip", RESULT_ROOT))
        drive_archive = RESULT_ROOT / "source_aware_sensitivity_results.zip"
        shutil.copy2(archive, drive_archive)
        print(f"archive: {drive_archive} ({drive_archive.stat().st_size / 1e9:.2f} GB)")
        '''),
        cell("code", r'''
        # 7) 下載結果；若瀏覽器阻擋大型檔案，可直接從上方 Drive 路徑取得
        from google.colab import files
        files.download("/content/source_aware_sensitivity_results.zip")
        '''),
        cell("markdown", r'''
        ## 請回傳

        將 `source_aware_sensitivity_results.zip` 提供給我。我會檢查 12 runs 是否完整，
        產生論文表格，並依預先定義的判讀規則撰寫 sensitivity analysis；不會事後挑最佳權重。
        '''),
    ]
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"name": OUTPUT.name, "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.x"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUTPUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
