# Source-Aware Sensitivity Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a directly executable Colab notebook that runs four predeclared source-aware arousal-weight configurations over three seeds and downloads a complete reproducibility bundle.

**Architecture:** A deterministic Python builder owns the notebook source, allowing the generated `.ipynb` to be reviewed and regenerated reliably. Repository tests inspect the notebook contract and execute extracted pure helper functions without requiring Colab, a GPU, or model training.

**Tech Stack:** Python 3, Jupyter notebook JSON, PyTorch, Transformers, pandas, NumPy, SciPy, Google Colab Drive, `unittest`.

## Global Constraints

- Configurations are exactly `uniform`, `mild`, `current`, and `reflection_swap`.
- Arousal weights are respectively `1/1/1/1`, `0.5/0.75/1/1`, `0.25/0.5/1/0.75`, and `0.25/0.5/0.75/1` for CVAS/CVAT/DSA-MST/EDU2021.
- Seeds are exactly 42, 1, and 2, yielding exactly 12 matched runs.
- Every run uses MacBERT, `l1_intensity`, batch size 32, four epochs, learning rate 2e-5, maximum length 256, and the existing checkpoint criterion.
- Every command passes the complete source-weight vector explicitly.
- No synthetic augmentation or ranking loss is used.
- Results must include per-run metrics, mean and sample SD, paired-seed changes from uniform, a LaTeX table, logs, predictions, checkpoints, and a manifest.
- An incomplete run matrix must fail loudly rather than generate a complete-looking summary.
- Preserve unrelated dirty-worktree changes and commit only files belonging to this plan.

---

## File Structure

- Create `notebooks/build_source_aware_sensitivity_notebook.py`: deterministic notebook builder and source-of-truth Colab cells.
- Create `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`: generated user-facing notebook.
- Create `tests/test_source_aware_sensitivity_notebook.py`: structural and pure-helper tests.
- Modify `docs/reproduce.md`: add the notebook and its returned artifact.

### Task 1: Notebook contract and experiment matrix

**Files:**
- Create: `tests/test_source_aware_sensitivity_notebook.py`
- Create: `notebooks/build_source_aware_sensitivity_notebook.py`
- Create: `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`

**Interfaces:**
- Produces: `build_notebook() -> dict` and `main() -> None` in the builder.
- Produces in notebook: `CONFIGS: dict[str, dict[str, float]]`, `SEEDS: list[int]`, and `expected_runs() -> list[tuple[str, int]]`.

- [ ] **Step 1: Write the failing structural test**

```python
class NotebookContractTests(unittest.TestCase):
    def test_exact_matrix(self):
        self.assertIn("SEEDS = [42, 1, 2]", self.source)
        for name in ("uniform", "mild", "current", "reflection_swap"):
            self.assertIn(f'"{name}"', self.source)
        self.assertIn('"sentence": 0.25', self.source)
        self.assertIn('"text": 0.50', self.source)
        self.assertIn('"reflection": 0.75', self.source)
        self.assertIn('"edu2021": 1.00', self.source)
```

- [ ] **Step 2: Verify the test fails**

Run: `python -m unittest tests.test_source_aware_sensitivity_notebook -v`

Expected: `FileNotFoundError` for the notebook.

- [ ] **Step 3: Implement the deterministic builder shell**

Implement `md(source)`, `code(source)`, `build_notebook()`, and `main()`. Generate Python 3/Colab metadata and initial title/configuration cells. `expected_runs()` must return config-major order with three seeds per configuration.

- [ ] **Step 4: Generate and test**

Run:

```bash
python notebooks/build_source_aware_sensitivity_notebook.py
python -m unittest tests.test_source_aware_sensitivity_notebook -v
python -m json.tool notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb >/dev/null
```

Expected: contract tests pass and notebook JSON is valid.

- [ ] **Step 5: Commit**

```bash
git add notebooks/build_source_aware_sensitivity_notebook.py \
  notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb \
  tests/test_source_aware_sensitivity_notebook.py
git commit -m "建立 source-aware sensitivity notebook 骨架"
```

### Task 2: Colab setup, preflight, and exact command construction

**Files:**
- Modify: `notebooks/build_source_aware_sensitivity_notebook.py`
- Modify: `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`
- Modify: `tests/test_source_aware_sensitivity_notebook.py`

**Interfaces:**
- Produces in notebook: `validate_inputs(repo_path: Path) -> dict[str, int]`.
- Produces in notebook: `source_weight_spec(config: str) -> str`.
- Produces in notebook: `build_train_command(config: str, seed: int) -> list[str]`.

- [ ] **Step 1: Add failing setup and command tests**

Extract pure definitions from the helper cell and assert:

```python
self.assertEqual(
    ns["source_weight_spec"]("current"),
    "sentence=1:0.25,text=1:0.5,reflection=1:1,edu2021=1:0.75",
)
cmd = ns["build_train_command"]("mild", 2)
self.assertIn("--source_aware", cmd)
self.assertEqual(cmd[cmd.index("--seed") + 1], "2")
self.assertEqual(cmd[cmd.index("--batch_size") + 1], "32")
self.assertEqual(cmd[cmd.index("--lex_mode") + 1], "l1_intensity")
self.assertNotIn("--rank_aug", cmd)
self.assertNotIn("--extra_train", cmd)
```

- [ ] **Step 2: Verify the new tests fail**

Run: `python -m unittest tests.test_source_aware_sensitivity_notebook -v`

Expected: missing helper functions and setup fragments.

- [ ] **Step 3: Add reproducible Colab setup and preflight cells**

Install bounded dependencies, require CUDA, mount Drive, clone or refresh the public repository, record its commit, and validate `train_v2.py`, training/dev CSV columns, lexicon files, and the four expected granularities. Print row counts and the 12-run matrix before training.

- [ ] **Step 4: Implement exact command helpers**

`build_train_command()` must start with `sys.executable, "train_v2.py"` and explicitly pass `--model`, `--lex_mode l1_intensity`, `--source_aware`, the full `--source_weights` value, run name, seed, epochs, batch size, learning rate, and max length.

- [ ] **Step 5: Regenerate, test, and commit**

Run the builder, unit test, and JSON validation commands from Task 1. Commit only the three task files with `git commit -m "加入 sensitivity 實驗預檢與命令建構"`.

### Task 3: Training lifecycle, resume, and run artifacts

**Files:**
- Modify: `notebooks/build_source_aware_sensitivity_notebook.py`
- Modify: `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`
- Modify: `tests/test_source_aware_sensitivity_notebook.py`

**Interfaces:**
- Produces in notebook: `run_name(config: str, seed: int) -> str`.
- Produces in notebook: `run_one(config: str, seed: int) -> dict`.
- Produces in notebook: `compute_dev_metrics(prediction_path: Path) -> dict[str, float]`.
- Produces in notebook: atomic `receipt.json` per completed run.

- [ ] **Step 1: Add failing lifecycle tests**

Assert run naming, streamed log capture, `os.replace` atomic receipt writing, SHA-256 recording, checkpoint/prediction existence checks, and metric computation from `valence_true`, `arousal_true`, `valence_pred`, and `arousal_pred`.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_source_aware_sensitivity_notebook -v`.

- [ ] **Step 3: Implement resumable run execution**

Execute commands sequentially while streaming output to the notebook and UTF-8 logs. A run is skipped only when its receipt, checkpoint, prediction CSV, hashes, config, and seed validate. Copy each completed run to Drive before writing its receipt atomically. Preserve checkpoints because the requested final ZIP includes them.

- [ ] **Step 4: Implement the 12-run loop**

Run all pairs from `expected_runs()`. On failure, record status, display the failed run, stop the matrix, and leave completed receipts reusable on the next execution.

- [ ] **Step 5: Regenerate, test, and commit**

Run builder/unit/JSON checks and commit with `git commit -m "實作 sensitivity 訓練續跑與產物保存"`.

### Task 4: Strict aggregation, LaTeX export, ZIP download, and docs

**Files:**
- Modify: `notebooks/build_source_aware_sensitivity_notebook.py`
- Modify: `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`
- Modify: `tests/test_source_aware_sensitivity_notebook.py`
- Modify: `docs/reproduce.md`

**Interfaces:**
- Produces in notebook: `collect_results(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`.
- Produces: `source_aware_per_run.csv`, `source_aware_summary.csv`, `source_aware_paired_vs_uniform.csv`, `source_aware_summary.tex`, `experiment_manifest.json`, and `source_aware_sensitivity_results.zip`.

- [ ] **Step 1: Add failing aggregation tests**

Create 12 temporary valid receipt fixtures. Assert 12 per-run rows, four summary rows, `ddof=1`, and nine non-uniform paired rows. Remove one receipt and assert `RuntimeError` contains `incomplete 12-run matrix`.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_source_aware_sensitivity_notebook -v`.

- [ ] **Step 3: Implement strict summaries and exports**

Validate exact config/seed pairs before writing any summary. Aggregate all four metrics by mean and sample SD. Compute signed within-seed differences from uniform, retaining the metric direction in column names. Generate a compact LaTeX table without selecting or bolding a winner.

- [ ] **Step 4: Implement the downloadable bundle**

Write package metadata, Python/package versions, git commit, configuration matrix, commands, timestamps, and artifact hashes to `experiment_manifest.json`. Archive tables, manifest, receipts, logs, predictions, and all 12 checkpoints. Show archive size, persist it to Drive, and call `google.colab.files.download`.

- [ ] **Step 5: Update reproduction documentation**

Add the notebook path, 4×3 matrix, resume behavior, approximate A100 requirement, and instruction to return `source_aware_sensitivity_results.zip`.

- [ ] **Step 6: Regenerate, test, and commit**

Run builder/unit/JSON checks and `git diff --check`. Commit the four planned files with `git commit -m "完成 source-aware sensitivity 結果匯出"`.

### Task 5: Final verification audit

**Files:**
- Verify all four deliverables from Tasks 1–4.

**Interfaces:**
- Consumes the completed notebook implementation; produces no new interface.

- [ ] **Step 1: Verify deterministic generation**

Run the builder twice and compare SHA-256 hashes. Expected: identical notebook hashes.

- [ ] **Step 2: Compile all ordinary Python code cells**

Parse or compile all code cells, excluding only Colab/IPython shell-magic lines. Expected: no syntax errors.

- [ ] **Step 3: Run final checks**

```bash
python -m unittest tests.test_source_aware_sensitivity_notebook -v
python -m json.tool notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb >/dev/null
git diff --check
git status --short
```

Expected: tests pass; JSON and whitespace validation pass; unrelated pre-existing changes remain unstaged.

- [ ] **Step 4: Inspect the cell sequence**

Print cell indices, types, and first lines. Confirm setup precedes preflight, training precedes aggregation, incomplete runs block exports, and the last cell downloads the ZIP.

- [ ] **Step 5: Commit audit fixes only if needed**

If corrections are necessary, stage only planned deliverables and commit with `git commit -m "修正 sensitivity notebook 驗證問題"`.

