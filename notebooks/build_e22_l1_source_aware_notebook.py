"""Generate the L1 x source-aware 2x2 ablation Colab notebook deterministically.

Why this notebook exists
------------------------
ROCLING-2026 reviewer 1 (submission #46) asked for the one missing cell of the
lexicon x source-aware 2x2 design: **L1 (10-d) + source-aware arousal weighting**
(the run historically numbered E22, which has never completed --- three local
`mps` attempts hung, see `docs/archive/handover.md` 3.6).

Without that cell the paper cannot separate "L1++ contributes nothing on its own
but becomes useful once source weighting is added" from "only source weighting
matters".  The notebook therefore runs all four cells under one protocol
(MacBERT, batch 32, lr 2e-5, 4 epochs, max_len 256, same checkpoint criterion,
seeds 42/1/2) so the comparison is not confounded by the historical batch-size
and seed-count differences between E4 (batch 64, 1 seed) and E18/E19.

|                      | no `--source_aware` | `--source_aware` |
|----------------------|---------------------|------------------|
| `--lex_mode l1`      | `e4_l1`             | **`e22_l1_sa`**  |
| `--lex_mode l1_intensity` | `e18_l1pp`     | `e19_l1pp_sa`    |

Properties of the generated notebook:
  * every deliverable is persisted to Google Drive with a hashed receipt, so a
    disconnected runtime never loses a finished run and re-running the main cell
    resumes instead of retraining;
  * each run emits official-format prediction CSVs for both the 200-text public
    validation split and the 1,100-text test split, and each CSV is zipped as
    `submission.csv.zip` (internal name `submission.csv`, exactly what the
    scoring site expects);
  * the dev proxy (253 labelled DSA-MST documents) is scored locally, so the
    2x2 table exists even if no leaderboard quota is available;
  * checkpoints (390 MB) are deleted after the Drive artifacts are verified.
"""
import json
import pathlib
import textwrap


HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE / "Rocling2026_Colab_e22_l1_source_aware.ipynb"


def cell(kind, source):
    return {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(source).lstrip("\n").splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


def build_notebook():
    cells = [
        cell("markdown", '''
        # ROCLING 2026：L1 x source-aware 2x2 ablation（含 reviewer 指名的 E22）

        審稿意見 1 指出：論文只有 E18（L1++）、E19（L1++ + source-aware）與 E4（L1）三格，
        缺 **L1 + source-aware**（= E22）這一格，因此無法判斷 E19 的增益是來自 L1++
        還是只來自 source-aware 監督。這份 notebook 補上該格，並在**同一協定**下重跑
        另外三格，避免拿 batch 64 單 seed 的 E4 去比 batch 32 的 E18/E19。

        | | 無 `--source_aware` | 有 `--source_aware` |
        |---|---|---|
        | `--lex_mode l1`（10 維） | `e4_l1` | **`e22_l1_sa`（reviewer 要的那格）** |
        | `--lex_mode l1_intensity`（31 維） | `e18_l1pp` | `e19_l1pp_sa` |

        **固定協定**：`hfl/chinese-macbert-base`、batch 32、lr 2e-5、4 epochs、max_len 256、
        checkpoint criterion = `mean(PCC) - mean(MAE)` on dev、seeds 42/1/2。
        4 conditions x 3 seeds = **12 runs**，A100 約 3 小時。

        **每個 run 的產出**
        - `dev_predictions.csv`（253 篇 DSA-MST，有 gold → 本機即可算四項指標）
        - `validation_submission.csv`（200 篇官方 public validation，官方格式）
        - `test_submission.csv`（1,100 篇官方 test，官方格式）
        - `validation_submission.csv.zip` / `test_submission.csv.zip`
          （壓縮檔內部檔名一律為 `submission.csv`，可直接上傳評分網站）

        另外在 Drive 根目錄會產生一份 `submission.csv.zip`，內容 = **E22 seed 42 的
        validation 預測**，也就是要補進論文 Table 4（official public validation）的那一列。

        **斷線後**：重跑前置 cells 與主迴圈即可，已完成的 run 會用 Drive receipt 跳過。
        '''),

        cell("code", '''
        # 1) 安裝套件並確認 GPU
        !pip -q install "transformers>=4.40" "huggingface_hub>=0.23" jieba scikit-learn scipy pandas

        import platform, subprocess
        import torch

        assert torch.cuda.is_available(), "沒有 CUDA GPU：請在 Colab 選 GPU runtime 後重跑。本機 mps 在此設定下會 hang。"
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {GPU_NAME} ({GPU_VRAM_GB:.1f} GB)")
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
        '''),

        cell("code", '''
        # 2) 掛載 Google Drive；所有可交付結果都持久化到這裡
        from google.colab import drive
        from pathlib import Path
        import os

        drive.mount("/content/drive")
        RESULT_ROOT = Path("/content/drive/MyDrive/ROCLING2026_l1_source_aware")
        RUNS_ROOT = RESULT_ROOT / "runs"
        SUBMIT_ROOT = RESULT_ROOT / "submissions"
        for d in (RESULT_ROOT, RUNS_ROOT, SUBMIT_ROOT):
            d.mkdir(parents=True, exist_ok=True)

        probe = RESULT_ROOT / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        assert probe.read_text(encoding="utf-8") == "ok"
        probe.unlink()
        print("結果目錄:", RESULT_ROOT)
        '''),

        cell("code", '''
        # 3) 匿名取得公開 repo；首次執行記錄 commit，續跑時固定使用同一 commit
        import json, subprocess
        from pathlib import Path

        REPO_URL = "https://github.com/chen0427ok/DSA-NIFT.git"
        REPO_PATH = Path("/content/DSA-NIFT")
        ENV_MANIFEST_PATH = RESULT_ROOT / "environment_manifest.json"
        old_manifest = json.loads(ENV_MANIFEST_PATH.read_text(encoding="utf-8")) if ENV_MANIFEST_PATH.exists() else {}
        pinned_repo_commit = old_manifest.get("repo_commit")

        if not (REPO_PATH / ".git").exists():
            subprocess.run(["git", "clone", "--branch", "main", REPO_URL, str(REPO_PATH)], check=True)
        actual_remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=REPO_PATH,
            check=True, capture_output=True, text=True
        ).stdout.strip()
        assert actual_remote.rstrip("/") == REPO_URL.rstrip("/"), f"非預期 remote: {actual_remote}"

        if pinned_repo_commit:
            subprocess.run(["git", "fetch", "origin", pinned_repo_commit], cwd=REPO_PATH, check=True)
            subprocess.run(["git", "checkout", "--detach", pinned_repo_commit], cwd=REPO_PATH, check=True)
        else:
            subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_PATH, check=True)
            subprocess.run(["git", "checkout", "--detach", "origin/main"], cwd=REPO_PATH, check=True)

        REPO_COMMIT = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_PATH,
            check=True, capture_output=True, text=True
        ).stdout.strip()
        if pinned_repo_commit:
            assert REPO_COMMIT == pinned_repo_commit, "repo commit 與既有實驗 manifest 不一致"
        os.chdir(REPO_PATH)
        print("Repo commit:", REPO_COMMIT)
        '''),

        cell("code", '''
        # 4) 固定同一個 MacBERT revision，並寫入環境 manifest
        import importlib.metadata as metadata
        from huggingface_hub import model_info, snapshot_download

        MODEL_ID = "hfl/chinese-macbert-base"
        MODEL_SHA = old_manifest.get("model_sha") or model_info(MODEL_ID).sha
        MODEL_DIR = Path("/content/model_snapshot")
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_SHA, local_dir=str(MODEL_DIR))

        BASE_MANIFEST = {
            "repo_url": REPO_URL,
            "repo_commit": REPO_COMMIT,
            "model_id": MODEL_ID,
            "model_sha": MODEL_SHA,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": metadata.version("transformers"),
            "huggingface_hub": metadata.version("huggingface_hub"),
            "gpu": GPU_NAME,
            "gpu_vram_gb": round(GPU_VRAM_GB, 2),
        }
        if old_manifest:
            assert old_manifest["repo_commit"] == REPO_COMMIT
            assert old_manifest["model_sha"] == MODEL_SHA
        tmp_manifest = ENV_MANIFEST_PATH.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(BASE_MANIFEST, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_manifest, ENV_MANIFEST_PATH)
        print(json.dumps(BASE_MANIFEST, ensure_ascii=False, indent=2))
        '''),

        cell("code", '''
        # 5) 唯一的 2x2 設定矩陣與 preflight
        import pandas as pd

        SEEDS = [42, 1, 2]
        BATCH_SIZE = 32
        LR = 2e-5
        MAX_LEN = 256
        EPOCHS = 4

        # True = 只跑 reviewer 指名的那一格（3 runs，約 45 分鐘）；
        # False = 跑完整 2x2（12 runs，約 3 小時，論文表格才不會被 batch size 混淆）。
        ONLY_MISSING_CELL = False
        DELETE_CHECKPOINT_AFTER_PERSIST = True   # 權重刪除後只能重訓
        STOP_ON_ERROR = False                    # 單一 run 失敗不中斷其餘 run
        FORCE_RERUN = False

        ALL_CONDITIONS = {
            "e4_l1":       {"lex_mode": "l1",           "source_aware": False, "paper_id": "E4"},
            "e18_l1pp":    {"lex_mode": "l1_intensity", "source_aware": False, "paper_id": "E18"},
            "e19_l1pp_sa": {"lex_mode": "l1_intensity", "source_aware": True,  "paper_id": "E19"},
            "e22_l1_sa":   {"lex_mode": "l1",           "source_aware": True,  "paper_id": "E22"},
        }
        CONDITIONS = ({"e22_l1_sa": ALL_CONDITIONS["e22_l1_sa"]} if ONLY_MISSING_CELL
                      else dict(ALL_CONDITIONS))

        # 補進論文 official validation 表的那一列 = E22 seed 42 的 validation 預測
        PRIMARY_CONDITION, PRIMARY_SEED, PRIMARY_SPLIT = "e22_l1_sa", 42, "validation"

        REQUIRED_FILES = [
            "train_v2.py", "predict.py", "lexicon.py", "lexicon_intensity.py",
            "data/train.csv", "data/dev.csv", "data/val_unlabeled.csv", "data/DSANIDF_TestSet.csv",
        ]
        missing = [p for p in REQUIRED_FILES if not Path(p).exists()]
        assert not missing, f"缺少必要檔案: {missing}"

        expected_columns = {
            "data/train.csv": {"id", "text", "valence", "arousal", "granularity"},
            "data/dev.csv": {"id", "text", "valence", "arousal"},
            "data/val_unlabeled.csv": {"id", "text"},
            "data/DSANIDF_TestSet.csv": {"id", "text"},
        }
        row_counts = {}
        for path, needed in expected_columns.items():
            frame = pd.read_csv(path)
            columns = {str(c).strip().lower() for c in frame.columns}
            assert needed <= columns, f"{path} 欄位錯誤: {frame.columns.tolist()}"
            assert len(frame) > 0, f"{path} 是空檔"
            row_counts[path] = len(frame)
        assert row_counts["data/val_unlabeled.csv"] == 200, row_counts
        assert row_counts["data/DSANIDF_TestSet.csv"] == 1100, row_counts

        # source-aware 的權重表靠 granularity 欄；缺值會被靜默當成 1:1，等於整個實驗失效
        granularity = pd.read_csv("data/train.csv")["granularity"].astype(str).str.strip()
        assert not granularity.isin(["", "nan"]).any(), "train.csv 有空的 granularity"
        counts = granularity.value_counts()
        assert {"sentence", "text", "reflection", "edu2021"} <= set(counts.index), counts
        print("granularity 分布:\\n", counts.to_string())

        matrix = pd.DataFrame([
            {"condition": c, "seed": s, "paper_id": cfg["paper_id"], "lex_mode": cfg["lex_mode"],
             "source_aware": cfg["source_aware"], "batch_size": BATCH_SIZE, "lr": LR,
             "epochs": EPOCHS, "max_len": MAX_LEN}
            for c, cfg in CONDITIONS.items() for s in SEEDS
        ])
        display(matrix)
        print("row counts:", row_counts)
        assert len(matrix) == len(CONDITIONS) * len(SEEDS)
        '''),

        cell("code", '''
        # 6) 訓練、推論、submission 打包、Drive receipt helpers
        import csv, hashlib, shutil, sys, time, zipfile
        import numpy as np

        def run_name(condition, seed):
            return f"cell2x2_{condition}_s{seed}"

        def sha256(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        def atomic_json(path, payload):
            path = Path(path)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)

        def execute_logged(command, log_path):
            print("$", " ".join(map(str, command)), flush=True)
            with open(log_path, "w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    list(map(str, command)), cwd=REPO_PATH,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                rc = process.wait()
            if rc != 0:
                raise RuntimeError(f"command failed (rc={rc}): {' '.join(map(str, command))}")

        def build_train_command(condition, seed):
            cfg = CONDITIONS[condition]
            cmd = [
                sys.executable, "train_v2.py",
                "--model", str(MODEL_DIR),
                "--run_name", run_name(condition, seed),
                "--lex_mode", cfg["lex_mode"],
                "--seed", str(seed),
                "--epochs", str(EPOCHS),
                "--batch_size", str(BATCH_SIZE),
                "--lr", str(LR),
                "--max_len", str(MAX_LEN),
            ]
            if cfg["source_aware"]:
                # 預設權重 = 論文 E19 的 CVAS .25 / CVAT .50 / DSA-MST 1.00 / EDU2021 .75
                cmd.append("--source_aware")
            return cmd

        def build_predict_command(condition, seed, ckpt):
            cfg = CONDITIONS[condition]
            return [
                sys.executable, "predict.py", "--ckpt", str(ckpt),
                "--model", str(MODEL_DIR), "--lex_mode", cfg["lex_mode"],
                "--input", "data/DSANIDF_TestSet.csv", "--run_name", run_name(condition, seed),
                "--split", "test", "--batch_size", str(BATCH_SIZE), "--max_len", str(MAX_LEN),
            ]

        def dev_metrics(path):
            df = pd.read_csv(path)
            result = {}
            for dim, prefix in [("valence", "V"), ("arousal", "A")]:
                gold = df[f"{dim}_true"].to_numpy(float)
                pred = df[f"{dim}_pred"].to_numpy(float)
                result[f"{prefix}_MAE"] = float(np.mean(np.abs(pred - gold)))
                result[f"{prefix}_PCC"] = float(np.corrcoef(pred, gold)[0, 1])
            return result

        def check_submission(csv_path, expected_rows):
            """官方格式把關：欄位、列數、無缺值、預測落在 [1, 9]。"""
            df = pd.read_csv(csv_path)
            assert list(df.columns) == ["ID", "Valence", "Arousal"], df.columns.tolist()
            assert len(df) == expected_rows, f"{csv_path}: {len(df)} 列，應為 {expected_rows}"
            assert df["ID"].is_unique and df.notna().all().all(), f"{csv_path}: ID 重複或有缺值"
            for col in ["Valence", "Arousal"]:
                values = df[col].to_numpy(float)
                assert values.min() >= 1.0 and values.max() <= 9.0, f"{csv_path}: {col} 超出 [1,9]"
            return df

        def zip_submission(csv_path, zip_path):
            """壓成評分網站要的 submission.csv.zip（內部檔名固定 submission.csv）。"""
            zip_path = Path(zip_path)
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(csv_path, arcname="submission.csv")
            with zipfile.ZipFile(zip_path) as zf:
                assert zf.namelist() == ["submission.csv"], zf.namelist()
            return zip_path

        def valid_receipt(run):
            run_dir = RUNS_ROOT / run
            receipt_path = run_dir / "receipt.json"
            if not receipt_path.exists():
                return False
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                for name, expected_hash in receipt["artifact_sha256"].items():
                    path = run_dir / name
                    if not path.exists() or path.stat().st_size == 0 or sha256(path) != expected_hash:
                        return False
                return receipt["status"] == "complete"
            except Exception as exc:
                print(f"receipt invalid for {run}: {exc}")
                return False

        def run_one(condition, seed):
            run = run_name(condition, seed)
            if valid_receipt(run) and not FORCE_RERUN:
                print(f"skip {run}: Drive receipt 已完成")
                return "skipped"

            cfg = CONDITIONS[condition]
            run_dir = RUNS_ROOT / run
            run_dir.mkdir(parents=True, exist_ok=True)
            local_log = Path("/content") / f"{run}.log"
            started = time.time()

            train_cmd = build_train_command(condition, seed)
            execute_logged(train_cmd, local_log)

            ckpt = REPO_PATH / "outputs" / f"{run}_best.pt"
            dev_pred = REPO_PATH / "outputs" / "preds" / f"{run}_dev.csv"
            val_pred = REPO_PATH / "outputs" / "preds" / f"{run}_val.csv"
            val_sub = REPO_PATH / "outputs" / f"{run}_submission.csv"
            for path in [ckpt, dev_pred, val_pred, val_sub]:
                assert path.exists() and path.stat().st_size > 0, f"缺少訓練產物: {path}"

            predict_cmd = build_predict_command(condition, seed, ckpt)
            execute_logged(predict_cmd, local_log.with_name(f"{run}_predict.log"))
            test_pred = REPO_PATH / "outputs" / "preds" / f"{run}_test.csv"
            test_sub = REPO_PATH / "outputs" / f"{run}_test_submission.csv"
            for path in [test_pred, test_sub]:
                assert path.exists() and path.stat().st_size > 0, f"缺少 test 產物: {path}"

            check_submission(val_sub, 200)
            check_submission(test_sub, 1100)
            val_zip = zip_submission(val_sub, Path("/content") / f"{run}_validation_submission.csv.zip")
            test_zip = zip_submission(test_sub, Path("/content") / f"{run}_test_submission.csv.zip")

            metrics = dev_metrics(dev_pred)
            config = {
                **BASE_MANIFEST, "run_name": run, "condition": condition, "seed": seed,
                "paper_id": cfg["paper_id"], "lex_mode": cfg["lex_mode"],
                "source_aware": cfg["source_aware"], "batch_size": BATCH_SIZE, "lr": LR,
                "epochs": EPOCHS, "max_len": MAX_LEN, "train_command": train_cmd,
                "predict_command": predict_cmd, "elapsed_minutes": (time.time() - started) / 60,
            }
            local_config = Path("/content") / f"{run}_config.json"
            local_metrics = Path("/content") / f"{run}_dev_metrics.json"
            local_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            local_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

            artifacts = {
                "train.log": local_log,
                "predict.log": local_log.with_name(f"{run}_predict.log"),
                "config.json": local_config,
                "dev_metrics.json": local_metrics,
                "dev_predictions.csv": dev_pred,
                "validation_predictions.csv": val_pred,
                "test_predictions.csv": test_pred,
                "validation_submission.csv": val_sub,
                "test_submission.csv": test_sub,
                "validation_submission.csv.zip": val_zip,
                "test_submission.csv.zip": test_zip,
            }
            for name, source in artifacts.items():
                shutil.copy2(source, run_dir / name)
            # 另外集中一份好上傳的副本：submissions/{run}_{split}_submission.csv.zip
            shutil.copy2(val_zip, SUBMIT_ROOT / f"{run}_validation_submission.csv.zip")
            shutil.copy2(test_zip, SUBMIT_ROOT / f"{run}_test_submission.csv.zip")

            artifact_hashes = {name: sha256(run_dir / name) for name in artifacts}
            receipt = {
                "status": "complete", "run_name": run, "condition": condition, "seed": seed,
                "paper_id": cfg["paper_id"], "dev_metrics": metrics,
                "artifact_sha256": artifact_hashes,
                "repo_commit": REPO_COMMIT, "model_sha": MODEL_SHA,
            }
            atomic_json(run_dir / "receipt.json", receipt)
            assert valid_receipt(run), f"Drive persistence verification failed: {run}"

            if DELETE_CHECKPOINT_AFTER_PERSIST:
                ckpt.unlink()
                print(f"已驗證 Drive artifacts，刪除 runtime checkpoint: {ckpt.name}")
            print(f"OK {run}: {metrics}")
            return "completed"
        '''),

        cell("code", '''
        # 7) 主迴圈。中斷後重跑本 cell 即可續跑（已完成的 run 會跳過）。
        # 先跑 reviewer 指名的 e22，確保即使 runtime 中途斷線，最關鍵的那一格已經到手。
        ordered_conditions = (["e22_l1_sa"] + [c for c in CONDITIONS if c != "e22_l1_sa"]
                              if "e22_l1_sa" in CONDITIONS else list(CONDITIONS))

        statuses = []
        for condition in ordered_conditions:
            for seed in SEEDS:
                run = run_name(condition, seed)
                print("\\n" + "=" * 80)
                print(">>", run, CONDITIONS[condition])
                print("=" * 80)
                try:
                    statuses.append({"run_name": run, "status": run_one(condition, seed)})
                except Exception as exc:
                    statuses.append({"run_name": run, "status": "failed", "error": repr(exc)})
                    print(f"FAILED {run}: {exc}")
                    if STOP_ON_ERROR:
                        raise
        display(pd.DataFrame(statuses))
        '''),

        cell("code", '''
        # 8) 彙整 dev 2x2、打包 submission.csv.zip、產生官方分數填寫表與 handoff zip
        records = []
        for condition in CONDITIONS:
            for seed in SEEDS:
                run = run_name(condition, seed)
                if not valid_receipt(run):
                    continue
                receipt = json.loads((RUNS_ROOT / run / "receipt.json").read_text(encoding="utf-8"))
                records.append({
                    "run_name": run, "condition": condition, "paper_id": receipt["paper_id"],
                    "lex_mode": CONDITIONS[condition]["lex_mode"],
                    "source_aware": CONDITIONS[condition]["source_aware"], "seed": seed,
                    **receipt["dev_metrics"], "repo_commit": receipt["repo_commit"],
                    "model_sha": receipt["model_sha"],
                })

        per_seed = (pd.DataFrame(records).sort_values(["condition", "seed"])
                    if records else pd.DataFrame())
        per_seed.to_csv(RESULT_ROOT / "dev_metrics_per_seed.csv", index=False)

        summary_rows = []
        for condition, cfg in CONDITIONS.items():
            part = per_seed[per_seed["condition"] == condition] if not per_seed.empty else pd.DataFrame()
            observed = set(part["seed"].astype(int)) if not part.empty else set()
            complete = observed == set(SEEDS)
            row = {"condition": condition, "paper_id": cfg["paper_id"], "lex_mode": cfg["lex_mode"],
                   "source_aware": cfg["source_aware"], "n_seed": len(observed),
                   "status": "complete" if complete else "incomplete"}
            for metric in ["V_MAE", "V_PCC", "A_MAE", "A_PCC"]:
                row[f"{metric}_mean"] = part[metric].mean() if complete else np.nan
                row[f"{metric}_sd"] = part[metric].std(ddof=1) if complete else np.nan
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(RESULT_ROOT / "dev_metrics_summary.csv", index=False)
        display(summary)

        # dev proxy 上的 2x2（arousal 是 reviewer 關心的維度）
        if not per_seed.empty and (summary["status"] == "complete").all():
            for metric in ["A_PCC", "A_MAE"]:
                pivot = summary.pivot(index="lex_mode", columns="source_aware",
                                      values=f"{metric}_mean")
                print(f"\\n=== dev {metric}（3-seed mean）===")
                print(pivot.round(4).to_string())
        else:
            print("\\n尚有 condition 未跑完，先不畫 2x2。")

        # 官方分數填寫表：每個 run x split 一列，上傳評分網站後把四項指標填回來
        score_rows = [
            {"run_name": run_name(c, s), "paper_id": CONDITIONS[c]["paper_id"], "condition": c,
             "seed": s, "split": split, "submission_zip": f"{run_name(c, s)}_{split}_submission.csv.zip",
             "V_MAE": "", "V_PCC": "", "A_MAE": "", "A_PCC": ""}
            for c in CONDITIONS for s in SEEDS for split in ["validation", "test"]
        ]
        scores_path = RESULT_ROOT / "official_scores_to_fill.csv"
        pd.DataFrame(score_rows).to_csv(scores_path, index=False)
        matrix.to_csv(RESULT_ROOT / "run_manifest.csv", index=False)

        # 主提交檔：reviewer 指名那一格的 validation 預測 -> Drive 根目錄 submission.csv.zip
        primary_run = run_name(PRIMARY_CONDITION, PRIMARY_SEED)
        primary_csv = RUNS_ROOT / primary_run / f"{PRIMARY_SPLIT}_submission.csv"
        if primary_csv.exists():
            check_submission(primary_csv, 200 if PRIMARY_SPLIT == "validation" else 1100)
            zip_submission(primary_csv, RESULT_ROOT / "submission.csv.zip")
            print(f"\\n主提交檔 submission.csv.zip <- {primary_run} ({PRIMARY_SPLIT})")
        else:
            print(f"\\n找不到 {primary_csv}，主提交檔未產生（先把 {primary_run} 跑完）。")

        local_handoff = Path("/content/l1_source_aware_handoff.zip")
        if local_handoff.exists():
            local_handoff.unlink()
        with zipfile.ZipFile(local_handoff, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in RESULT_ROOT.rglob("*"):
                if path.is_file() and path.name != "l1_source_aware_handoff.zip":
                    zf.write(path, arcname=str(path.relative_to(RESULT_ROOT)))
        shutil.copy2(local_handoff, RESULT_ROOT / "l1_source_aware_handoff.zip")

        print("Handoff:", RESULT_ROOT / "l1_source_aware_handoff.zip")
        print("可上傳的 zip：", sorted(p.name for p in SUBMIT_ROOT.glob("*.zip")))
        print(f"完成 receipts: {len(records)}/{len(CONDITIONS) * len(SEEDS)}")
        '''),

        cell("markdown", '''
        ## 跑完之後

        ### 1. 上傳評分網站（若還有額度）
        Drive 根目錄的 `submission.csv.zip` 就是 **E22 seed 42 的 public validation 預測**，
        直接上傳即可補上論文 Table 4 缺的那一列。其餘 zip 在 `submissions/`，
        壓縮檔內部檔名都是 `submission.csv`。

        額度有限時的建議順序（先拿最能回答審稿意見的分數）：

        1. `cell2x2_e22_l1_sa_s42_validation_submission.csv.zip` ← **最重要**
        2. `cell2x2_e22_l1_sa_s42_test_submission.csv.zip`
        3. `cell2x2_e4_l1_s42_*`（同協定的 L1 對照，讓 2x2 不被 batch size 混淆）
        4. `cell2x2_e18_l1pp_s42_*`、`cell2x2_e19_l1pp_sa_s42_*`
        5. 其餘 seeds（1、2），用來估 seed 變異

        分數填進 `official_scores_to_fill.csv` 的四個欄位。

        ### 2. 沒有額度也不會卡住
        `dev_metrics_per_seed.csv` / `dev_metrics_summary.csv` 是 253 篇 DSA-MST dev proxy
        的四項指標（有 gold，本機算得出來），已經足以回答審稿意見的核心問題：
        **拿掉 L1++、只留 source-aware，arousal 還剩多少增益。**
        論文須註明這是 source-domain proxy，不是官方 target-domain 分數。

        ### 3. 把結果變成論文表格
        把 `l1_source_aware_handoff.zip`（或單獨的兩個 csv）下載回本機後：

        ```bash
        python baseline/make_l1_source_aware_table.py \\
            --dev_summary <解壓路徑>/dev_metrics_summary.csv \\
            --dev_per_seed <解壓路徑>/dev_metrics_per_seed.csv \\
            --official <解壓路徑>/official_scores_to_fill.csv   # 有官方分數才加
        ```

        會覆寫 `paper/tables/l1_source_aware_2x2.tex`，
        論文 `\\input` 該檔的段落（Section 5「Disentangling ...」）即可直接編譯。
        '''),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
