# Batch-controlled Ablation Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a focused Colab notebook that runs four batch-32 MacBERT conditions over three seeds and exports reproducible artifacts for removing batch-confounding language from the paper after official scores are returned.

**Architecture:** A small Python builder produces a deterministic `.ipynb` with self-contained Colab cells for setup, immutable model resolution, preflight checks, training, inference, Drive-backed receipts, summaries, and handoff packaging. A repository test parses the generated notebook and exercises extracted pure helper definitions without running GPU training.

**Tech Stack:** Python 3, Jupyter notebook JSON (`nbformat`), PyTorch, Transformers, Hugging Face Hub, pandas, NumPy, Google Colab Drive, `unittest`.

## Global Constraints

- Clone public `https://github.com/chen0427ok/DSA-NIFT.git` from branch `main` without credentials.
- Conditions are exactly `nolex_b32`, `l1_b32`, `aug_E_b32`, and `aug_F2_b32`.
- Seeds are exactly 42, 1, and 2.
- All runs use batch size 32, learning rate 2e-5, maximum length 256, and four epochs.
- All runs use one resolved revision of `hfl/chinese-macbert-base` and the checkpoint criterion already implemented in `train_v2.py`.
- A run is resumable only after its small artifacts are verified on Drive and an atomic completion receipt exists.
- Never silently summarize fewer than three seeds as a complete condition.
- Do not edit `paper/main.tex` until the user returns official leaderboard scores.
- Preserve unrelated dirty-worktree changes and commit only files belonging to each task.

---

## File Structure

- Create `notebooks/build_batch_controlled_ablation_notebook.py`: deterministic notebook builder and source-of-truth cell content.
- Create `notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb`: generated user-facing Colab notebook.
- Create `tests/test_batch_controlled_notebook.py`: structural and pure-helper tests that do not need a GPU or network.
- Modify `docs/reproduce.md`: add the new notebook and explain the result handoff.

### Task 1: Notebook contract and deterministic builder shell

**Files:**
- Create: `tests/test_batch_controlled_notebook.py`
- Create: `notebooks/build_batch_controlled_ablation_notebook.py`
- Create: `notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb`

**Interfaces:**
- Produces: `build_notebook() -> dict`, a notebook JSON object with `nbformat == 4`.
- Produces: `main() -> None`, writing the notebook beside the builder.
- Consumes: no project training code; this task establishes only the notebook contract.

- [ ] **Step 1: Write the failing structural test**

```python
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Rocling2026_Colab_batch_controlled_ablation.ipynb"


class NotebookContractTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.source = "\n".join(
            "".join(cell.get("source", [])) for cell in self.nb["cells"]
        )

    def test_notebook_metadata_and_conditions(self):
        self.assertEqual(self.nb["nbformat"], 4)
        self.assertEqual(self.nb["metadata"]["kernelspec"]["name"], "python3")
        for name in ("nolex_b32", "l1_b32", "aug_E_b32", "aug_F2_b32"):
            self.assertIn(name, self.source)

    def test_controlled_hyperparameters_and_seeds_are_explicit(self):
        for fragment in (
            "SEEDS = [42, 1, 2]", "BATCH_SIZE = 32", "LR = 2e-5",
            "MAX_LEN = 256", "EPOCHS = 4",
        ):
            self.assertIn(fragment, self.source)

    def test_public_clone_has_no_token_prompt(self):
        self.assertIn("https://github.com/chen0427ok/DSA-NIFT.git", self.source)
        self.assertNotIn("getpass", self.source)
        self.assertNotIn("github token", self.source.lower())
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m unittest tests.test_batch_controlled_notebook -v`

Expected: `FileNotFoundError` for the not-yet-created notebook.

- [ ] **Step 3: Implement the deterministic builder shell**

Implement `md(source: str) -> dict`, `code(source: str) -> dict`, and
`build_notebook() -> dict`. Use `textwrap.dedent`, split sources with
`keepends=True`, set Colab-compatible Python 3 metadata, and write JSON with
`ensure_ascii=False, indent=1`. The initial notebook must contain its title,
the four-condition table, the public clone URL, and a configuration cell with:

```python
SEEDS = [42, 1, 2]
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 256
EPOCHS = 4
CONDITIONS = {
    "nolex_b32": {"lex_mode": "none", "extra_train": None},
    "l1_b32": {"lex_mode": "l1", "extra_train": None},
    "aug_E_b32": {"lex_mode": "l1", "extra_train": "data/train_aug_E.csv"},
    "aug_F2_b32": {"lex_mode": "l1", "extra_train": "data/train_aug_F2.csv"},
}
```

- [ ] **Step 4: Generate the notebook and rerun the structural test**

Run:

```bash
python notebooks/build_batch_controlled_ablation_notebook.py
python -m unittest tests.test_batch_controlled_notebook -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the notebook shell**

```bash
git add notebooks/build_batch_controlled_ablation_notebook.py \
  notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb \
  tests/test_batch_controlled_notebook.py
git commit -m "建立 batch-controlled Colab notebook 骨架"
```

### Task 2: Reproducible setup, model pinning, and preflight

**Files:**
- Modify: `notebooks/build_batch_controlled_ablation_notebook.py`
- Modify: `notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb`
- Modify: `tests/test_batch_controlled_notebook.py`

**Interfaces:**
- Produces in notebook: `REPO_PATH`, `RESULT_ROOT`, `MODEL_DIR`, `MODEL_SHA`, and `BASE_MANIFEST`.
- Produces in notebook: `validate_inputs() -> dict[str, int]`, returning row counts keyed by required CSV path.
- Consumes: condition constants from Task 1.

- [ ] **Step 1: Extend tests for setup and preflight contracts**

Add assertions that notebook source contains:

```python
for fragment in (
    "git clone --depth 1 --branch main",
    "model_info(MODEL_ID).sha",
    "snapshot_download(repo_id=MODEL_ID, revision=MODEL_SHA",
    "drive.mount('/content/drive')",
    "def validate_inputs()",
    "train_v2.py",
    "predict.py",
    "data/train_aug_E.csv",
    "data/train_aug_F2.csv",
    "data/DSANIDF_TestSet.csv",
):
    self.assertIn(fragment, self.source)
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run: `python -m unittest tests.test_batch_controlled_notebook.NotebookContractTests -v`

Expected: failures for missing setup and preflight fragments.

- [ ] **Step 3: Add Colab setup cells**

Add cells that install pinned minimum dependencies, fail when CUDA is absent,
clone `main` into `/content/DSA-NIFT`, record `git rev-parse HEAD`, mount Drive,
and create `/content/drive/MyDrive/ROCLING2026_batch_controlled`. The clone cell
must delete no user data: if the directory exists, use `git fetch origin main`
and `git reset --hard origin/main` only inside the disposable Colab clone after
validating its remote URL.

- [ ] **Step 4: Add immutable model resolution and preflight**

Resolve `MODEL_SHA = model_info(MODEL_ID).sha`, download that exact revision to
`/content/model_snapshot`, and pass this local path to every training and
prediction command. Implement `validate_inputs()` to assert required files,
required `ID`/`Text` columns for inference files, label columns for training and
dev files, non-empty CSVs, and distinct augmentation paths. Print row counts and
the full experiment matrix before training.

- [ ] **Step 5: Generate and test**

Run:

```bash
python notebooks/build_batch_controlled_ablation_notebook.py
python -m unittest tests.test_batch_controlled_notebook -v
python -m json.tool notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb >/dev/null
```

Expected: all tests pass and JSON validation exits 0.

- [ ] **Step 6: Commit setup and preflight**

```bash
git add notebooks/build_batch_controlled_ablation_notebook.py \
  notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb \
  tests/test_batch_controlled_notebook.py
git commit -m "加入 Colab 固定環境與資料預檢"
```

### Task 3: Training, inference, persistence, and safe resume

**Files:**
- Modify: `notebooks/build_batch_controlled_ablation_notebook.py`
- Modify: `notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb`
- Modify: `tests/test_batch_controlled_notebook.py`

**Interfaces:**
- Produces in notebook: `run_name(condition: str, seed: int) -> str`.
- Produces in notebook: `build_train_command(condition: str, seed: int) -> list[str]`.
- Produces in notebook: `run_one(condition: str, seed: int) -> dict`.
- Produces in notebook: `atomic_write_json(path: Path, payload: dict) -> None`.
- Consumes: `MODEL_DIR`, `RESULT_ROOT`, condition constants, and validated inputs from Task 2.

- [ ] **Step 1: Add pure-helper and persistence contract tests**

Extract the notebook's helper-definition cell with `ast.parse` and execute only
imports, assignments, function definitions, and class definitions in a test
namespace. Assert:

```python
self.assertEqual(ns["run_name"]("l1_b32", 42), "controlled_l1_b32_s42")
cmd = ns["build_train_command"]("aug_E_b32", 1)
self.assertEqual(cmd[cmd.index("--batch_size") + 1], "32")
self.assertEqual(cmd[cmd.index("--seed") + 1], "1")
self.assertEqual(cmd[cmd.index("--lex_mode") + 1], "l1")
self.assertEqual(cmd[cmd.index("--extra_train") + 1], "data/train_aug_E.csv")
self.assertNotIn("--extra_train", ns["build_train_command"]("l1_b32", 1))
```

Also assert the notebook includes `os.replace`, receipt validation, `tee`-style
log capture, test inference, artifact copy verification, and a checkpoint-delete
guard controlled by `DELETE_CHECKPOINT_AFTER_PERSIST`.

- [ ] **Step 2: Run tests and verify the new assertions fail**

Run: `python -m unittest tests.test_batch_controlled_notebook -v`

Expected: helper names or required persistence fragments are missing.

- [ ] **Step 3: Implement exact command construction**

`build_train_command` must return a list beginning with `sys.executable,
"train_v2.py"` and explicitly include `--model MODEL_DIR`, `--run_name`,
`--lex_mode`, `--seed`, `--epochs 4`, `--batch_size 32`, `--lr 2e-5`, and
`--max_len 256`. Append `--extra_train` only for E and F2.

- [ ] **Step 4: Implement run lifecycle**

For each run, check a validated Drive receipt first. Otherwise run training with
stdout/stderr streamed to screen and a UTF-8 log, assert the checkpoint and
dev/val files exist, then call `predict.py` on `data/DSANIDF_TestSet.csv` with
the matching local model path and lexicon mode. Calculate four dev metrics from
the saved predictions. Package validation and test CSVs into separate zip files
with the internal name `submission.csv`.

- [ ] **Step 5: Implement verified persistence and cleanup**

Copy logs, predictions, raw submissions, submission zips, run configuration,
and dev metrics to a temporary Drive run directory. Verify file sizes and SHA-256
hashes, atomically rename the directory, then atomically write `receipt.json`.
Delete the local checkpoint only when the receipt validates and
`DELETE_CHECKPOINT_AFTER_PERSIST` is true. A failed command raises and leaves no
receipt. The outer loop records the failure and stops by default so a partially
failed matrix is visible.

- [ ] **Step 6: Generate and run tests**

Run:

```bash
python notebooks/build_batch_controlled_ablation_notebook.py
python -m unittest tests.test_batch_controlled_notebook -v
python -m json.tool notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb >/dev/null
```

Expected: all tests pass.

- [ ] **Step 7: Commit training and resume flow**

```bash
git add notebooks/build_batch_controlled_ablation_notebook.py \
  notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb \
  tests/test_batch_controlled_notebook.py
git commit -m "實作 controlled ablation 訓練與續跑"
```

### Task 4: Summary, leaderboard handoff, and documentation

**Files:**
- Modify: `notebooks/build_batch_controlled_ablation_notebook.py`
- Modify: `notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb`
- Modify: `tests/test_batch_controlled_notebook.py`
- Modify: `docs/reproduce.md`

**Interfaces:**
- Produces in notebook: `collect_results(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]`.
- Produces: `controlled_ablation_handoff.zip` and `official_scores_to_fill.csv`.
- Consumes: valid receipts and per-run metrics from Task 3.

- [ ] **Step 1: Add summary and archive tests**

Create temporary receipt fixtures for all 12 runs. Assert
`collect_results()` returns 12 per-seed rows and four summary rows, uses sample
standard deviation (`ddof=1`), and marks each condition complete only when seeds
equal `{42, 1, 2}`. Add a missing-seed fixture and assert it raises
`RuntimeError("incomplete condition")`. Assert the score-entry columns are:

```python
["run_name", "condition", "seed", "V_MAE", "V_PCC", "A_MAE", "A_PCC"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m unittest tests.test_batch_controlled_notebook -v`

Expected: `collect_results` is missing.

- [ ] **Step 3: Implement strict aggregation and handoff packaging**

Read only validated receipts. Write `run_manifest.csv`,
`dev_metrics_per_seed.csv`, and a mean/sample-SD `dev_metrics_summary.csv`.
Generate `official_scores_to_fill.csv` with 12 rows and blank official metric
fields. Build `controlled_ablation_handoff.zip` from manifests, receipts, logs,
predictions, and all submission zips; exclude model checkpoints. Print the Drive
path and exact instruction to send either the archive or completed score CSV back.

- [ ] **Step 4: Update reproduction documentation**

Add the notebook to the notebook table in `docs/reproduce.md`, describe its
12-run controlled matrix, anonymous public clone, Drive resume behavior, and
the required returned artifact. Do not change historical experiment commands.

- [ ] **Step 5: Generate and run full verification**

Run:

```bash
python notebooks/build_batch_controlled_ablation_notebook.py
python -m unittest discover -s tests -v
python -m json.tool notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb >/dev/null
git diff --check
```

Expected: all tests pass, notebook JSON is valid, and no whitespace errors are
reported.

- [ ] **Step 6: Commit summary and docs**

```bash
git add notebooks/build_batch_controlled_ablation_notebook.py \
  notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb \
  tests/test_batch_controlled_notebook.py docs/reproduce.md
git commit -m "完成 controlled ablation 結果交付流程"
```

### Task 5: Final notebook audit

**Files:**
- Verify: `notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb`
- Verify: `notebooks/build_batch_controlled_ablation_notebook.py`
- Verify: `tests/test_batch_controlled_notebook.py`
- Verify: `docs/reproduce.md`

**Interfaces:**
- Consumes all deliverables from Tasks 1–4.
- Produces no new interface; this is the completion gate.

- [ ] **Step 1: Verify builder determinism**

Run the builder twice and compare the notebook hash:

```bash
python notebooks/build_batch_controlled_ablation_notebook.py
shasum -a 256 notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb
python notebooks/build_batch_controlled_ablation_notebook.py
shasum -a 256 notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb
```

Expected: identical hashes.

- [ ] **Step 2: Compile code cells without executing Colab operations**

Run a checker that removes lines beginning with `%` or `!`, replaces the body
of cells containing Colab-only top-level operations with parsed helper
definitions, and calls `compile(source, cell_name, "exec")` for every code cell.

Expected: every code cell compiles.

- [ ] **Step 3: Run the final automated checks**

Run:

```bash
python -m unittest discover -s tests -v
python -m json.tool notebooks/Rocling2026_Colab_batch_controlled_ablation.ipynb >/dev/null
git diff --check
git status --short
```

Expected: tests pass; JSON and whitespace checks pass; status contains no
unexpected changes from this implementation.

- [ ] **Step 4: Inspect the rendered cell sequence**

Print each cell index, type, and first line. Confirm setup precedes training,
training precedes summary, destructive checkpoint deletion is opt-in and
guarded, and the final cell states what the user should return.

- [ ] **Step 5: Commit audit fixes only if required**

If the audit required a correction, stage only the four planned deliverables
and commit with:

```bash
git commit -m "修正 controlled ablation notebook 驗證問題"
```
