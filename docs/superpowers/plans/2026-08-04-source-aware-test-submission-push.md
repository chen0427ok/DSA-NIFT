# Source-Aware Test Submission Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the source-aware sensitivity Colab notebook to generate, validate, commit, and safely push 12 official test submissions to `origin/main`.

**Architecture:** Add pure, locally testable helpers for test-inference commands, submission validation, destination mapping, and publication preconditions to the deterministic notebook builder. Keep external Git authentication and push execution in a separate opt-in Colab cell that consumes only already-validated Drive artifacts.

**Tech Stack:** Python 3, Jupyter notebook JSON, pandas, NumPy, Git CLI, Google Colab, `unittest`.

## Global Constraints

- Produce exactly 12 files under `test_submissions/`, one for every predeclared configuration/seed pair.
- Each file must have exactly 1,100 rows, exact columns `ID,Valence,Arousal`, ordered IDs matching `data/DSANIDF_TestSet.csv`, unique non-empty IDs, and finite values in `[1,9]`.
- Test inference must use `predict.py`, the exact run checkpoint, `l1_intensity`, the official test CSV, `--split test`, batch size 32, and max length 256.
- A receipt without checkpoint, test predictions, and validated test submission hashes is invalid.
- Stage only the 12 explicit submission paths and use commit message `加入 source-aware sensitivity test predictions`.
- Push only with `git push origin HEAD:main`; never force-push, reset, or rewrite remote history.
- Token material must not be stored in the notebook, repository origin URL, logs, receipts, or manifest.
- Preserve unrelated dirty-worktree changes and commit only files belonging to this plan.

---

### Task 1: Test inference and submission validation

**Files:**
- Modify: `tests/test_source_aware_sensitivity_notebook.py`
- Modify: `notebooks/build_source_aware_sensitivity_notebook.py`
- Generate: `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`

**Interfaces:**
- Produces in notebook: `build_test_command(config: str, seed: int, checkpoint: Path, out_dir: Path) -> list[str]`.
- Produces in notebook: `test_submission_name(config: str, seed: int) -> str`.
- Produces in notebook: `validate_test_submission(path: Path, test_input: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing command, naming, and validation tests**

```python
def test_test_command_uses_matching_checkpoint_and_official_input(self):
    cmd = self.ns["build_test_command"]("current", 42, Path("model.pt"), Path("out"))
    self.assertEqual(cmd[1], "predict.py")
    self.assertEqual(cmd[cmd.index("--ckpt") + 1], "model.pt")
    self.assertEqual(cmd[cmd.index("--input") + 1], "data/DSANIDF_TestSet.csv")
    self.assertEqual(cmd[cmd.index("--split") + 1], "test")
    self.assertEqual(cmd[cmd.index("--lex_mode") + 1], "l1_intensity")

def test_submission_validation_rejects_wrong_id_order(self):
    with self.assertRaisesRegex(ValueError, "ID sequence"):
        self.ns["validate_test_submission"](submission, official_test)
```

Add separate fixtures for wrong columns, row count, duplicate/empty IDs,
non-finite values, and values outside `[1,9]`. Use a `required_rows` optional
argument defaulting to 1,100 so three-row fixtures exercise real validation.

- [ ] **Step 2: Run tests and verify they fail because helpers are missing**

Run: `python -m unittest tests.test_source_aware_sensitivity_notebook -v`

- [ ] **Step 3: Implement the pure helpers**

`test_submission_name("current", 42)` returns
`sa_sensitivity_current_s42_test_submission.csv`. The 12 calls over
`expected_runs()` must be unique. `validate_test_submission` reads both CSVs,
performs every global-constraint check, and returns row count plus SHA-256 only
after all checks pass.

- [ ] **Step 4: Regenerate and verify**

```bash
python notebooks/build_source_aware_sensitivity_notebook.py
python -m unittest tests.test_source_aware_sensitivity_notebook -v
python -m json.tool notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb >/dev/null
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_source_aware_sensitivity_notebook.py \
  notebooks/build_source_aware_sensitivity_notebook.py \
  notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb
git commit -m "加入 sensitivity test 推論與驗證"
```

### Task 2: Receipt-integrated inference and safe authentication

**Files:**
- Modify the same three files from Task 1.

**Interfaces:**
- Produces in notebook: `required_run_artifacts() -> tuple[str, ...]`.
- Produces in notebook: `temporary_askpass(token: str)` context manager yielding a Git environment mapping.
- Modifies: `run_one(config, seed)` to generate and persist test artifacts before receipt creation.

- [ ] **Step 1: Add failing receipt and token-safety tests**

```python
def test_receipt_requires_test_artifacts(self):
    self.assertIn("test_predictions.csv", self.ns["required_run_artifacts"]())
    self.assertIn("test_submission.csv", self.ns["required_run_artifacts"]())

def test_notebook_does_not_embed_or_persist_token(self):
    self.assertNotIn("x-access-token:", self.source)
    self.assertIn("GIT_ASKPASS", self.source)
    self.assertIn("finally", self.source)
```

- [ ] **Step 2: Verify the tests fail**

Run the targeted unit suite and confirm failures name the missing artifact and
unsafe clone URL behavior.

- [ ] **Step 3: Integrate test inference into `run_one`**

After training, copy the checkpoint into the run directory, execute
`build_test_command` against that copied checkpoint, validate the generated
submission, rename outputs to `test_predictions.csv` and
`test_submission.csv`, and include both in the receipt. `valid_receipt` must
require every name from `required_run_artifacts()` and revalidate the test CSV.

- [ ] **Step 4: Replace tokenized clone URL with temporary askpass**

Prompt once with `getpass`, create a mode-`0700` temporary askpass script, pass
credentials only through its environment, sanitize `origin` to `REPO_URL`, and
delete the helper in `finally`. Retain the token only in the live Colab variable
needed by the publication cell.

- [ ] **Step 5: Regenerate, test, and commit**

Run builder/unit/JSON checks and commit the three files with
`git commit -m "整合 test 產物與安全 Git 認證"`.

### Task 3: Atomic 12-file commit and non-force push

**Files:**
- Modify: `tests/test_source_aware_sensitivity_notebook.py`
- Modify: `notebooks/build_source_aware_sensitivity_notebook.py`
- Generate: `notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb`
- Modify: `docs/reproduce.md`

**Interfaces:**
- Produces in notebook: `publication_paths(result_root: Path, repo_path: Path) -> list[tuple[Path, Path]]`.
- Produces in notebook: `validate_staged_paths(staged: list[str], expected: list[Path]) -> None`.
- Produces a separate opt-in cell controlled by `PUSH_TO_MAIN = False` by default.

- [ ] **Step 1: Add failing publication-contract tests**

Assert 12 unique repository destinations, rejection of 11 sources, rejection
of unexpected staged files, absence of `--force`/`-f`, exact
`["git", "push", "origin", "HEAD:main"]`, exact commit message, and default
`PUSH_TO_MAIN = False`.

- [ ] **Step 2: Verify the tests fail**

Run the unit suite and confirm publication helpers/cell are missing.

- [ ] **Step 3: Implement publication preflight and cell**

The cell validates all receipts/submissions, displays source/destination pairs,
and exits before Git mutation unless the user edits `PUSH_TO_MAIN = True`. It
fetches `origin main`, allows only a clean fast-forward, rejects differing
existing remote files, copies files, stages exact paths, validates the staged
set, commits only if changed, and pushes `origin HEAD:main` through askpass.
Push rejection preserves the local commit and raises a clear non-force error.

- [ ] **Step 4: Update execution documentation**

Document the new test inference outputs, explicit opt-in publication cell,
single commit, target `main`, token handling, and non-force conflict behavior.

- [ ] **Step 5: Run final verification**

```bash
python notebooks/build_source_aware_sensitivity_notebook.py
python -m unittest discover -s tests -v
python -m json.tool notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb >/dev/null
git diff --check
```

Compile every ordinary Python cell after excluding IPython magic lines and run
the builder twice to confirm identical SHA-256 hashes.

- [ ] **Step 6: Commit**

```bash
git add docs/reproduce.md tests/test_source_aware_sensitivity_notebook.py \
  notebooks/build_source_aware_sensitivity_notebook.py \
  notebooks/Rocling2026_Colab_source_aware_sensitivity.ipynb
git commit -m "加入 test submissions 安全推送流程"
```

