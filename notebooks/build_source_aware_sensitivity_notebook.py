"""Generate the source-aware sensitivity Colab notebook deterministically.

Design: docs/superpowers/specs/2026-08-05-source-aware-sensitivity-rewrite-design.md

Key properties of the generated notebook:
  * no Google Drive, no ZIP archive, no files.download — the repository is the
    storage layer and every run is pushed as soon as it finishes;
  * checkpoints (390 MB each) stay on Colab local disk and are deleted after
    test inference, so nothing large is ever hashed, copied or downloaded;
  * a failed run never aborts the remaining runs.
"""
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

        固定矩陣的 sensitivity check，**不是** hyperparameter search：矩陣事先定死，
        所有 configuration 都會回報，不會看完結果再挑最好的。

        | config | CVAS `sentence` | CVAT `text` | DSA-MST `reflection` | `edu2021` |
        |---|---:|---:|---:|---:|
        | `uniform` | 1.00 | 1.00 | 1.00 | 1.00 |
        | `mild` | 0.50 | 0.75 | 1.00 | 1.00 |
        | `current` | 0.25 | 0.50 | 1.00 | 0.75 |
        | `reflection_swap` | 0.25 | 0.50 | 0.75 | 1.00 |

        權重只作用在 arousal；valence 對所有真實來源固定 1.0。

        ## 怎麼跑

        依序執行 cell 1→6。**中斷了就重跑 cell 5**：已完成的 run 會從 repo 讀到
        receipt 並跳過。每個 run 一跑完就 commit + push，所以結果不會因為 session
        掛掉而消失。

        實測 T4 上每個 run 約 31 分鐘，12 個要 6 小時以上。要分段跑就設 cell 5 的
        `RUN_ONLY`，例如 `["current"]` 只跑 current 的三個 seed。

        390 MB 的 checkpoint 只留在 Colab 本地碟，test 推論一做完就刪除；
        Drive 沒有掛載，也不會產生 ZIP 或瀏覽器下載。
        '''),

        cell("code", r'''
        # 1) 安裝依賴並確認 GPU
        %pip install -q "transformers>=4.40,<5" "pandas>=2" "numpy>=1.24" jieba
        import torch
        assert torch.cuda.is_available(), "請在 Runtime > Change runtime type 選 GPU 後重跑"
        print("GPU:", torch.cuda.get_device_name(0),
              f"({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB)")
        print("torch:", torch.__version__)
        '''),

        cell("code", r'''
        # 2) 取得 repo 並「先」驗證 push 權限（避免訓練兩小時後才發現推不上去）
        import contextlib, getpass, os, pathlib, subprocess, tempfile

        REPO_URL = "https://github.com/chen0427ok/DSA-NIFT.git"
        REPO_PATH = pathlib.Path("/content/DSA-NIFT")

        @contextlib.contextmanager
        def git_auth_env(token):
            """把 token 只放進 subprocess env 與臨時 askpass helper，絕不寫進 remote URL。"""
            env = os.environ.copy()
            helper = None
            if token:
                fd, helper = tempfile.mkstemp(prefix="rocling-askpass-")
                os.close(fd)
                pathlib.Path(helper).write_text(
                    '#!/bin/sh\ncase "$1" in *Username*) printf "%s\\n" "x-access-token" ;; '
                    '*) printf "%s\\n" "$ROCLING_GITHUB_TOKEN" ;; esac\n', encoding="utf-8")
                os.chmod(helper, 0o700)
                env.update({"GIT_ASKPASS": helper, "GIT_TERMINAL_PROMPT": "0",
                            "ROCLING_GITHUB_TOKEN": token})
            try:
                yield env
            finally:
                if helper:
                    pathlib.Path(helper).unlink(missing_ok=True)

        # repo 是公開的，clone 不需要 token
        if not (REPO_PATH / ".git").exists():
            subprocess.run(["git", "clone", "--branch", "main", REPO_URL, str(REPO_PATH)],
                           check=True)
        subprocess.run(["git", "remote", "set-url", "origin", REPO_URL],
                       cwd=REPO_PATH, check=True)
        os.chdir(REPO_PATH)

        # push 需要 token（fine-grained PAT，Contents: Read and write）
        GITHUB_TOKEN = getpass.getpass("GitHub push token: ").strip()
        if not GITHUB_TOKEN:
            raise RuntimeError("需要 token 才能把每個 run 的結果 push 回 repo")

        with git_auth_env(GITHUB_TOKEN) as git_env:
            subprocess.run(["git", "fetch", "origin", "main"],
                           cwd=REPO_PATH, check=True, env=git_env)
            # 只做 fast-forward：絕不用 reset --hard，以免毀掉還沒推上去的 run
            sync = subprocess.run(["git", "merge", "--ff-only", "origin/main"],
                                  cwd=REPO_PATH, capture_output=True, text=True)
            if sync.returncode:
                print("注意：本地有尚未推送的 commit，publish 時會先 rebase 再 push")
            probe = subprocess.run(["git", "push", "--dry-run", "origin", "HEAD:main"],
                                   cwd=REPO_PATH, env=git_env,
                                   capture_output=True, text=True)
        if probe.returncode != 0:
            raise RuntimeError("push 權限驗證失敗，請檢查 token 的 Contents 權限；"
                               "git 回報：" + (probe.stderr.strip().splitlines() or ["(無訊息)"])[-1])
        print("push 權限 OK")

        for key, value in (("user.name", "%an"), ("user.email", "%ae")):
            identity = subprocess.check_output(["git", "log", "-1", f"--format={value}"],
                                               cwd=REPO_PATH, text=True).strip()
            subprocess.run(["git", "config", key, identity], cwd=REPO_PATH, check=True)

        REPO_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=REPO_PATH, text=True).strip()
        print("repo:", REPO_PATH, REPO_COMMIT)
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
        SOURCES = ("sentence", "text", "reflection", "edu2021")
        MODEL_ID = "hfl/chinese-macbert-base"
        METRICS = ["V_MAE", "V_PCC", "A_MAE", "A_PCC"]
        RESULT_SUBDIR = "results/source_aware_sensitivity"
        TEST_INPUT = "data/DSANIDF_TestSet.csv"
        TEST_ROWS = 1100
        CHECKPOINT_DIR = "outputs"

        # 改成 True 就把 inverted（完全反轉的 falsification control）加進矩陣，
        # 變成 5 configs × 3 seeds = 15 runs。其他地方都不用動。
        ENABLE_INVERTED = False

        BASE_CONFIGS = {
            "uniform":         {"sentence": 1.00, "text": 1.00, "reflection": 1.00, "edu2021": 1.00},
            "mild":            {"sentence": 0.50, "text": 0.75, "reflection": 1.00, "edu2021": 1.00},
            "current":         {"sentence": 0.25, "text": 0.50, "reflection": 1.00, "edu2021": 0.75},
            "reflection_swap": {"sentence": 0.25, "text": 0.50, "reflection": 0.75, "edu2021": 1.00},
        }
        INVERTED_CONFIG = {
            "inverted":        {"sentence": 1.00, "text": 0.75, "reflection": 0.50, "edu2021": 0.25},
        }

        def build_configs(enable_inverted):
            configs = dict(BASE_CONFIGS)
            if enable_inverted:
                configs.update(INVERTED_CONFIG)
            return configs

        CONFIGS = build_configs(ENABLE_INVERTED)
        REPO_PATH = pathlib.Path(globals().get("REPO_PATH", "."))

        def expected_runs():
            return [(config, seed) for config in CONFIGS for seed in SEEDS]

        def run_name(config, seed):
            return f"sa_sensitivity_{config}_s{seed}"

        def run_relative_dir(config, seed):
            return f"{RESULT_SUBDIR}/{run_name(config, seed)}"

        def run_dir(config, seed):
            return REPO_PATH / run_relative_dir(config, seed)

        def run_artifact_names():
            return ("receipt.json", "dev_predictions.csv", "test_submission.csv", "train.log")

        def run_relative_paths(config, seed):
            base = run_relative_dir(config, seed)
            return [f"{base}/{name}" for name in run_artifact_names()]

        def checkpoint_path(config, seed):
            return REPO_PATH / CHECKPOINT_DIR / f"{run_name(config, seed)}_best.pt"

        def _fmt_weight(value):
            return f"{value:g}"

        def source_weight_spec(config):
            weights = CONFIGS[config]
            return ",".join(f"{src}=1:{_fmt_weight(weights[src])}" for src in SOURCES)

        # -u 是必要的：子行程的 stdout 是 pipe 而非 tty，Python 會用 8KB 區塊緩衝，
        # 而整個 run 的 stdout 只有 ~1.5 KB，沒有 -u 就會到訓練結束才一次吐出，
        # 看起來像當掉。（bufsize=1 只影響父行程這端，救不了子行程。）
        def build_train_command(config, seed):
            return [sys.executable, "-u", "train_v2.py", "--model", MODEL_ID,
                    "--lex_mode", "l1_intensity", "--source_aware",
                    "--source_weights", source_weight_spec(config),
                    "--run_name", run_name(config, seed), "--seed", str(seed),
                    "--epochs", "4", "--batch_size", "32", "--lr", "2e-05",
                    "--max_len", "256"]

        def build_test_command(config, seed, checkpoint, out_dir):
            return [sys.executable, "-u", "predict.py", "--ckpt", str(checkpoint),
                    "--input", TEST_INPUT, "--run_name", run_name(config, seed),
                    "--split", "test", "--model", MODEL_ID,
                    "--lex_mode", "l1_intensity", "--out_dir", str(out_dir),
                    "--batch_size", "32", "--max_len", "256"]

        def commit_message(config, seed):
            return f"sa-sensitivity: {config} s{seed} 結果"

        def git_push_command():
            return ["git", "push", "origin", "HEAD:main"]

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
                out[f"{prefix}_PCC"] = (float(np.corrcoef(y, yhat)[0, 1])
                                        if np.std(yhat) > 1e-12 else 0.0)
            return out

        def validate_test_submission(path, test_input, required_rows=TEST_ROWS):
            frame = pd.read_csv(path, dtype={"ID": str})
            if list(frame.columns) != ["ID", "Valence", "Arousal"]:
                raise ValueError("columns must be exactly ID,Valence,Arousal")
            if len(frame) != required_rows:
                raise ValueError(f"row count must be {required_rows}, got {len(frame)}")
            ids = frame["ID"].astype(str)
            if frame["ID"].isna().any() or ids.str.strip().eq("").any():
                raise ValueError("IDs must be non-empty")
            if ids.duplicated().any():
                raise ValueError("IDs must be unique")
            expected_ids = pd.read_csv(test_input, dtype={"ID": str})["ID"].astype(str).tolist()
            if ids.tolist() != expected_ids:
                raise ValueError("ID sequence does not match official test input")
            values = frame[["Valence", "Arousal"]].apply(pd.to_numeric,
                                                         errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("predictions must be finite")
            if ((values < 1) | (values > 9)).any():
                raise ValueError("predictions must be in range [1,9]")
            return {"rows": len(frame), "sha256": sha256(path)}

        def validate_staged_paths(staged, expected):
            wanted = {pathlib.Path(p).as_posix() for p in expected}
            actual = {pathlib.Path(p).as_posix() for p in staged}
            if not actual.issubset(wanted):
                raise RuntimeError("unexpected staged paths: "
                                   f"{sorted(actual - wanted)}; expected subset of {sorted(wanted)}")

        def missing_runs(rows):
            done = {(r["config"], int(r["seed"])) for r in rows}
            return [pair for pair in expected_runs() if pair not in done]

        def aggregate_rows(rows):
            missing = missing_runs(rows)
            if missing:
                raise RuntimeError(f"incomplete matrix; missing={missing}")
            order = {name: i for i, name in enumerate(CONFIGS)}
            frame = pd.DataFrame(rows)
            frame["_order"] = frame["config"].map(order)
            frame = (frame.sort_values(["_order", "seed"])
                          .drop(columns="_order").reset_index(drop=True))
            summary_rows = []
            for config in CONFIGS:
                group = frame[frame.config == config]
                row = {"config": config, "n_seeds": len(group)}
                for metric in METRICS:
                    row[f"{metric}_mean"] = group[metric].mean()
                    row[f"{metric}_std"] = group[metric].std(ddof=1)
                summary_rows.append(row)
            base = frame[frame.config == "uniform"].set_index("seed")
            paired_rows = []
            for config in [c for c in CONFIGS if c != "uniform"]:
                for _, row in frame[frame.config == config].iterrows():
                    paired = {"config": config, "seed": int(row.seed)}
                    for metric in METRICS:
                        paired[f"delta_{metric}_vs_uniform"] = (
                            row[metric] - base.loc[row.seed, metric])
                    paired_rows.append(paired)
            return frame, pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)

        def summary_latex(summary):
            head = [r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
                    "Config & " + " & ".join(m.replace("_", "-") for m in METRICS) + r" \\",
                    r"\midrule"]
            body = []
            for _, row in summary.iterrows():
                cells = [f"{row[f'{m}_mean']:.4f} $\\pm$ {row[f'{m}_std']:.4f}" for m in METRICS]
                body.append(row["config"].replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
            return "\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n"

        def validate_inputs(repo_path):
            repo_path = pathlib.Path(repo_path)
            required = ["train_v2.py", "predict.py", TEST_INPUT, "data/train.csv",
                        "data/dev.csv", "external/emobank/CVAW_all_SD.csv",
                        "external/emobank/CVAP_all_SD.csv"]
            missing = [p for p in required if not (repo_path / p).exists()]
            if missing:
                raise FileNotFoundError("missing required files: " + ", ".join(missing))
            counts = {}
            for name in ("train", "dev"):
                frame = pd.read_csv(repo_path / f"data/{name}.csv")
                need = {"id", "granularity", "text", "valence", "arousal"}
                if not need.issubset(frame.columns):
                    raise ValueError(f"data/{name}.csv missing {sorted(need - set(frame.columns))}")
                counts[name] = len(frame)
                if name == "train" and not set(SOURCES).issubset(set(frame.granularity)):
                    raise ValueError(f"train.csv granularities incomplete: "
                                     f"{sorted(set(frame.granularity))}")
            test_rows = len(pd.read_csv(repo_path / TEST_INPUT))
            if test_rows != TEST_ROWS:
                raise ValueError(f"{TEST_INPUT} must have {TEST_ROWS} rows, got {test_rows}")
            counts["test"] = test_rows
            return counts

        print(validate_inputs(REPO_PATH))
        print(f"{len(CONFIGS)} configs × {len(SEEDS)} seeds = {len(expected_runs())} runs")
        display(pd.DataFrame([{"config": c, "seed": s, "source_weights": source_weight_spec(c)}
                              for c, s in expected_runs()]))
        '''),

        cell("code", r'''
        # 4) 單一 run：訓練 → test 推論 → 驗證 → 寫檔 → commit & push → 刪 checkpoint
        def atomic_write_json(path, payload):
            path = pathlib.Path(path)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)

        def git(*args, env=None, check=True):
            return subprocess.run(["git", *args], cwd=REPO_PATH, env=env, check=check,
                                  capture_output=True, text=True)

        def already_done(config, seed):
            """已完成判定只看 repo 裡的小檔案，不碰 checkpoint。"""
            directory = run_dir(config, seed)
            receipt_path = directory / "receipt.json"
            if not receipt_path.exists():
                return None
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["config"] != config or int(receipt["seed"]) != seed:
                    return None
                if not all((directory / name).exists() for name in run_artifact_names()):
                    return None
                validate_test_submission(directory / "test_submission.csv",
                                         REPO_PATH / TEST_INPUT)
                return receipt
            except (KeyError, ValueError, OSError, json.JSONDecodeError):
                return None

        def run_streamed(command, log_path, header):
            """即時印出並落地成 log。

            子行程一定要帶 -u（見 build_train_command），否則 stdout 會被 8KB 區塊
            緩衝，整個 run 到結束才吐輸出。有 -u 之後訓練每 50 steps 就有一行。
            例外：HF 首次下載 MacBERT 的進度條用 \\r 不換行，這裡的逐行迭代讀不到，
            所以第一個 run 開頭仍會有 1-2 分鐘沒有輸出，那是在下載模型。
            """
            print(f"--- {header} ---", flush=True)
            with pathlib.Path(log_path).open("a", encoding="utf-8") as log:
                log.write(f"\n===== {header} =====\n$ {' '.join(command)}\n")
                process = subprocess.Popen(command, cwd=REPO_PATH, text=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, bufsize=1)
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)

        def publish_run(config, seed):
            """commit 本 run 的 4 個小檔並 non-force push；被拒就 rebase 後重試一次。"""
            paths = run_relative_paths(config, seed)
            git("add", "--", *paths)
            staged = git("diff", "--cached", "--name-only").stdout.splitlines()
            validate_staged_paths(staged, paths)
            if not staged:
                return "unchanged"
            git("commit", "-m", commit_message(config, seed))
            with git_auth_env(GITHUB_TOKEN) as env:
                for attempt in (1, 2):
                    git("fetch", "origin", "main", env=env)
                    rebase = git("rebase", "origin/main", check=False)
                    if rebase.returncode:
                        git("rebase", "--abort", check=False)
                        raise RuntimeError("rebase onto origin/main failed: "
                                           + rebase.stderr.strip())
                    pushed = git(*git_push_command()[1:], env=env, check=False)
                    if pushed.returncode == 0:
                        return "pushed"
                    if attempt == 2:
                        raise RuntimeError("push rejected twice; 本地 commit 已保留，"
                                           "下次重跑 cell 5 會再試：" + pushed.stderr.strip())

        def run_one(config, seed):
            name = run_name(config, seed)
            done = already_done(config, seed)
            if done:
                print(f"SKIP {name}（repo 已有完整結果）")
                return done

            directory = run_dir(config, seed)
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "train.log"
            log_path.write_text("", encoding="utf-8")
            checkpoint = checkpoint_path(config, seed)
            test_out = pathlib.Path("/content/sa_test_out") / name
            try:
                run_streamed(build_train_command(config, seed), log_path, f"train {name}")
                dev_source = REPO_PATH / CHECKPOINT_DIR / "preds" / f"{name}_dev.csv"
                for source in (checkpoint, dev_source):
                    if not source.exists():
                        raise FileNotFoundError(f"training did not create {source}")
                shutil.copy2(dev_source, directory / "dev_predictions.csv")

                if test_out.exists():
                    shutil.rmtree(test_out)
                test_out.mkdir(parents=True)
                run_streamed(build_test_command(config, seed, checkpoint, test_out),
                             log_path, f"test inference {name}")
                submission = test_out / f"{name}_test_submission.csv"
                if not submission.exists():
                    raise FileNotFoundError(f"predict.py did not create {submission}")
                shutil.copy2(submission, directory / "test_submission.csv")

                test_validation = validate_test_submission(directory / "test_submission.csv",
                                                           REPO_PATH / TEST_INPUT)
                metrics = compute_dev_metrics(directory / "dev_predictions.csv")
                receipt = {
                    "run_name": name, "config": config, "seed": seed,
                    "weights": CONFIGS[config], "metrics": metrics,
                    "test_submission": test_validation,
                    "train_command": build_train_command(config, seed),
                    "repo_commit": REPO_COMMIT, "model": MODEL_ID,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "artifacts": [
                        {"name": n, "bytes": (directory / n).stat().st_size,
                         "sha256": sha256(directory / n)}
                        for n in run_artifact_names() if n != "receipt.json"
                    ],
                }
                atomic_write_json(directory / "receipt.json", receipt)
                print(f"[{name}] publish:", publish_run(config, seed))
                return receipt
            finally:
                # checkpoint 一律不保留：390 MB × 12 = 4.7 GB，且固定 seed 可重現
                checkpoint.unlink(missing_ok=True)
                shutil.rmtree(test_out, ignore_errors=True)
        '''),

        cell("code", r'''
        # 5) 主迴圈：失敗不中斷，修好後重跑本 cell 即可續跑
        #
        # 實測每個 run 約 31 分鐘，跑滿 12 個要 6 小時以上，單一 Colab session 很可能撐不住。
        # 想分段跑就填 RUN_ONLY，例如 ["current"] 只跑 current 的三個 seed；
        # [] 代表整個矩陣。無論怎麼填，repo 裡已完成的 run 都會自動跳過。
        RUN_ONLY = []
        MINUTES_PER_RUN = 31  # 觀測值，只用來估時

        selected = [(c, s) for c, s in expected_runs() if not RUN_ONLY or c in RUN_ONLY]
        if not selected:
            raise ValueError(f"RUN_ONLY={RUN_ONLY} 沒有對應任何 config；可選：{list(CONFIGS)}")
        todo = [(c, s) for c, s in selected if already_done(c, s) is None]
        print(f"選定 {len(selected)} runs：{len(selected) - len(todo)} 個已完成、{len(todo)} 個待跑")
        print(f"預估 {len(todo) * MINUTES_PER_RUN} 分鐘"
              f"（約 {len(todo) * MINUTES_PER_RUN / 60:.1f} 小時）")
        for config, seed in todo:
            print("  待跑:", run_name(config, seed))

        statuses = []
        for config, seed in selected:
            print(f"\n{'=' * 70}\n>>> {run_name(config, seed)}\n{'=' * 70}")
            try:
                receipt = run_one(config, seed)
                statuses.append({"config": config, "seed": seed, "status": "complete",
                                 **receipt["metrics"]})
            except Exception as exc:
                print(f"!!! {run_name(config, seed)} FAILED: {exc!r}")
                statuses.append({"config": config, "seed": seed, "status": "FAILED",
                                 "error": repr(exc)})

        status_frame = pd.DataFrame(statuses)
        display(status_frame)
        failed = status_frame[status_frame.status == "FAILED"]
        print(f"\n完成 {len(status_frame) - len(failed)}/{len(status_frame)} runs")
        if len(failed):
            print("失敗的 run（修正後重跑本 cell，已完成的會自動跳過）：")
            display(failed[["config", "seed", "error"]])
        '''),

        cell("code", r'''
        # 6) 彙整（只讀 repo 裡的 receipt，不重跑推論）→ commit & push
        rows = []
        for config, seed in expected_runs():
            receipt = already_done(config, seed)
            if receipt:
                rows.append({"config": config, "seed": seed, **receipt["metrics"]})

        missing = missing_runs(rows)
        display(pd.DataFrame(rows))
        if missing:
            print(f"矩陣不完整，缺 {len(missing)} 個 run：{missing}")
            print("未寫出 summary 檔案。請先重跑 cell 5 補齊。")
        else:
            per_run, summary, paired = aggregate_rows(rows)
            result_root = REPO_PATH / RESULT_SUBDIR
            outputs = {
                "source_aware_per_run.csv": per_run,
                "source_aware_summary.csv": summary,
                "source_aware_paired_vs_uniform.csv": paired,
            }
            for name, frame in outputs.items():
                frame.to_csv(result_root / name, index=False)
            (result_root / "source_aware_summary.tex").write_text(
                summary_latex(summary), encoding="utf-8")
            display(summary); display(paired)

            paths = [f"{RESULT_SUBDIR}/{n}" for n in
                     [*outputs, "source_aware_summary.tex"]]
            git("add", "--", *paths)
            staged = git("diff", "--cached", "--name-only").stdout.splitlines()
            validate_staged_paths(staged, paths)
            if staged:
                git("commit", "-m", "sa-sensitivity: 彙整 4 configs × 3 seeds 結果")
                with git_auth_env(GITHUB_TOKEN) as env:
                    git("fetch", "origin", "main", env=env)
                    git("rebase", "origin/main")
                    git(*git_push_command()[1:], env=env)
                print("彙整結果已 push 到 origin/main")
            else:
                print("彙整結果與 origin/main 相同，未建立重複 commit")
        '''),

        cell("markdown", r'''
        ## 跑完之後

        結果已經在 repo 的 `results/source_aware_sensitivity/`，不需要下載任何東西。

        撰寫論文時要遵守**事先講好**的判讀規則，不要看完結果再挑最佳權重：

        - `mild` ≈ `current` → 定性排序在起作用，精確數值不重要
        - `current` ≈ `reflection_swap` → DSA-MST 優於 EDU2021 的嚴格排序不重要
        - 全部落在 seed 變異範圍內 → 只能宣稱「權重選擇在此範圍內不敏感」
        - 差異超過 seed 變異 → 才能對 ordering 提出實證主張

        主要指標是 dev A-MAE / A-PCC；V-MAE / V-PCC 是 sanity check（valence 權重沒動，
        但共享參數仍可能間接影響）。

        **必須一併寫進論文的限制**：dev split 本身就抽自 DSA-MST，所以這個分析既不能
        證明獨立的 target-domain 泛化，也不是完整的 hyperparameter search。
        '''),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
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
