# Source-Aware Weight Sensitivity Colab — Rewrite Design

**Date:** 2026-08-05
**Status:** Design approved
**Supersedes:** `2026-08-04-source-aware-sensitivity-colab-design.md` and
`2026-08-04-source-aware-test-submission-push-design.md`

## Background

The 2026-08-04 notebook (`Rocling2026_Colab_source_aware_sensitivity.ipynb`, still
present at commit `3c8832e`) was executed in Colab but stalled, and no results
reached the repository. Post-mortem of that notebook identified four defects.

### Defect 1 — multi-gigabyte browser download

`run_one` copied each run's 390 MB checkpoint to Google Drive, so twelve runs
wrote 4.68 GB. Cell 6 then archived the entire Drive result root with
`shutil.make_archive`, producing a roughly 4.5 GB ZIP, and cell 8 passed that
archive to `google.colab.files.download`. That API streams through the browser
websocket and does not survive gigabyte-scale payloads. This is the most likely
stall point.

### Defect 2 — repeated SHA-256 over Drive FUSE

`valid_receipt` hashed every recorded artifact, including the 390 MB checkpoint,
reading it back through the Drive FUSE mount at roughly 10–30 MB/s. It ran once
per run inside `run_one` and again for all twelve runs in cell 6, re-reading
4.68 GB with no cell output during the scan.

### Defect 3 — Drive quota exhaustion

A free Drive account holds 15 GB. Writing 4.68 GB of checkpoints can exhaust the
quota, and the FUSE mount blocks rather than failing cleanly when it does.

### Defect 4 — silent blocking prompt

Cell 2 called `getpass.getpass()` unconditionally. The repository is now public
and needs no token to clone, but under *Runtime > Run all* the hidden input box
is easy to miss and blocks the whole notebook.

### Why the results were lost

Separately from the stall: cell 7 defaulted to `PUSH_TO_MAIN = False` and
required twelve valid receipts before acting. Until cells 6, 7, and 8 all
completed, every result existed only in the Colab session and in Drive. Cell 5
also re-raised on the first failure, abandoning all remaining runs.

## Objective

Rewrite the notebook as a predeclared robustness check of the four arousal
source weights used by E19. It is not a hyperparameter search: the matrix is
fixed in advance, every configuration is reported, and no configuration is
selected after inspecting results.

## Experimental Matrix

Weights apply to arousal only. Valence remains 1.0 for every real source.
Granularity keys match the `granularity` column of `data/train.csv`.

| Config | `sentence` (CVAS) | `text` (CVAT) | `reflection` (DSA-MST) | `edu2021` | Question addressed |
|---|---:|---:|---:|---:|---|
| `uniform` | 1.00 | 1.00 | 1.00 | 1.00 | Does source-aware weighting do anything at all? |
| `mild` | 0.50 | 0.75 | 1.00 | 1.00 | Does a gentler ordering preserve the behavior? Is EDU2021 at 1.0 acceptable? |
| `current` | 0.25 | 0.50 | 1.00 | 0.75 | Reproduce the submitted E19 setting. |
| `reflection_swap` | 0.25 | 0.50 | 0.75 | 1.00 | Does the DSA-MST over EDU2021 ordering matter? |

Each configuration runs with seeds 42, 1, and 2, giving twelve runs.

An `inverted` configuration (1.00 / 0.75 / 0.50 / 0.25) is defined next to
`CONFIGS` behind `ENABLE_INVERTED = False`. Setting that flag to `True` adds it
to the matrix as a falsification control without any other edit.

## Controlled Training Setup

Every run uses the existing `train_v2.py` and holds these fixed:

- encoder `hfl/chinese-macbert-base`;
- `--lex_mode l1_intensity` (31 dimensions);
- batch size 32, 4 epochs, learning rate 2e-5, maximum length 256;
- seeds 42, 1, 2;
- identical `data/train.csv` (9,435 rows) and `data/dev.csv` (253 reflection rows);
- no synthetic augmentation and no ranking loss.

All four weights are passed explicitly through `--source_weights` on every run.
The notebook never relies on `DEFAULT_SOURCE_WEIGHTS`. Run names are
`sa_sensitivity_{config}_s{seed}`.

## Architecture: the Repository Is the Storage Layer

Google Drive is not used. Results are written into the cloned repository working
tree and pushed after each run, so resume state lives in GitHub and survives a
dead Colab session, a new machine, or a later day.

```text
results/source_aware_sensitivity/<run_name>/
    receipt.json          ~2 KB    config, seed, weights, metrics, repo commit, timestamp
    dev_predictions.csv  ~30 KB    253 rows with gold labels
    test_submission.csv  ~50 KB    1,100 rows in official format
    train.log            ~10 KB
```

Each run stores roughly 90 KB, so the full matrix adds about 1 MB. `outputs/` is
already listed in `.gitignore`, so a 390 MB checkpoint cannot be committed even
by mistake.

Checkpoints live only on Colab local disk at
`/content/DSA-NIFT/outputs/<run_name>_best.pt` and are deleted once the run's
test inference finishes. Peak local disk use is one checkpoint. Reproduction is
by rerunning a fixed seed, not by retaining weights.

## Notebook Structure

| Cell | Purpose |
|---|---|
| 1 | Install dependencies, assert CUDA availability, print the GPU model |
| 2 | Clone the public repository without a token; prompt for a push token; verify push access immediately with `git ls-remote` |
| 3 | Pure helpers, the configuration table, and preflight file/data validation |
| 4 | `run_one` and its supporting functions |
| 5 | Main loop over the twelve runs |
| 6 | Aggregation, LaTeX table, and a final commit and push |

Cell 2 must fail fast. An invalid token or missing push permission has to surface
before any training begins, not after two hours of compute.

Preflight in cell 3 verifies that `train_v2.py`, `predict.py`,
`data/train.csv`, `data/dev.csv`, `data/DSANIDF_TestSet.csv`,
`external/emobank/CVAW_all_SD.csv`, and `external/emobank/CVAP_all_SD.csv` exist;
that `train.csv` contains all four granularity values; and that the test set has
exactly 1,100 rows.

## Per-Run Procedure

`run_one(config, seed)` performs these steps in order:

1. Skip when `results/source_aware_sensitivity/<run>/receipt.json` already exists
   and the stored `test_submission.csv` passes validation.
2. Train through `train_v2.py`, writing the checkpoint to local `outputs/`.
3. Run `predict.py` with `--input data/DSANIDF_TestSet.csv --split test` into a
   temporary directory under `/content`.
4. Validate the submission.
5. Recompute V-MAE, V-PCC, A-MAE, and A-PCC from `dev_predictions.csv`.
6. Write the four small artifacts and `receipt.json` into the repository.
7. Rebase onto `origin/main`, stage exactly the run's paths, reject unexpected
   staged entries, commit, and push with `git push origin HEAD:main`.
8. Delete the checkpoint.

Receipts record SHA-256 digests only for the small artifacts. Checkpoints are
never hashed.

The commit message is `sa-sensitivity: <config> s<seed> 結果`. Commit messages
must not mention a Claude co-author.

## Submission Validation Contract

Every `test_submission.csv` must satisfy all of the following:

- the header is exactly `ID,Valence,Arousal`;
- it contains exactly 1,100 data rows;
- IDs are non-empty and unique;
- the ID sequence matches `data/DSANIDF_TestSet.csv` element by element;
- Valence and Arousal are finite and lie within the inclusive range 1–9.

A run that fails validation produces no receipt and is retried on the next pass.

## Failure Handling

A failure at any step records the run's status, deletes the checkpoint, and
continues to the next run. The loop no longer re-raises. After the loop, a status
table lists every run with its outcome and, for failures, the error.

Pushes never use force. If a push is rejected because the remote advanced, the
notebook retries once after rebasing; a second rejection is reported and the
local commit is preserved for the next pass.

## Aggregation Outputs

Cell 6 reads only the committed receipts and does not rerun inference. It writes
into `results/source_aware_sensitivity/`:

- `source_aware_per_run.csv` — one row per configuration and seed;
- `source_aware_summary.csv` — mean and sample standard deviation (`ddof=1`) per configuration;
- `source_aware_paired_vs_uniform.csv` — within-seed differences against `uniform`;
- `source_aware_summary.tex` — a booktabs table for the paper.

With fewer than twelve complete runs the cell displays the partial table, names
the missing runs, and refuses to write the summary files. There is no ZIP archive
and no `files.download` call.

## Analysis Rules, Fixed in Advance

Primary outcomes are DSA-MST development A-MAE and A-PCC. V-MAE and V-PCC are
secondary sanity checks, since valence weights are unchanged but shared
parameters allow indirect effects.

Reporting uses the mean and sample standard deviation over the three predeclared
seeds, plus paired within-seed differences from `uniform`. Interpretation follows
these rules, chosen before seeing results:

- `mild` close to `current` indicates that the qualitative ordering, not the exact
  values, drives behavior;
- `current` close to `reflection_swap` indicates that the strict DSA-MST over
  EDU2021 ordering is unimportant;
- all configurations within seed-level variation supports only the narrow claim
  that the weight choice is insensitive within this range;
- a difference that exceeds seed variation is the only case that supports an
  empirical claim about the ordering.

A weight of 1.00 for DSA-MST is a reference value meaning "not down-weighted,"
not a claim of perfect reliability. It is anchored by matching task format,
document-level granularity, Traditional Chinese, first-person reflective style,
and the same 1–9 VA scale. EDU2021 shares the reflective style but has much
shorter texts, which motivates the original 0.75 without proving it.

The paper must state, wherever these results appear, that the development split
is itself drawn from DSA-MST, so this analysis establishes neither independent
target-domain generalization nor a complete hyperparameter search.

## Implementation Approach

Following the existing repository pattern, the notebook is generated by
`notebooks/build_source_aware_sensitivity_notebook.py`, and
`tests/test_source_aware_sensitivity_notebook.py` covers the pure helpers.

## Verification Criteria

The rewrite is complete when:

1. every ordinary Python cell compiles and the notebook is deterministic JSON;
2. generated commands cover exactly four configurations times three seeds, and
   enabling `ENABLE_INVERTED` yields fifteen;
3. each training command fixes the full four-source weight vector and every
   controlled argument;
4. each prediction command references the correct checkpoint, test input, run
   name, split, output directory, and lexicon mode;
5. submission validation rejects wrong headers, wrong row counts, empty or
   duplicate IDs, mismatched ID sequences, non-finite values, and out-of-range
   values;
6. staged-path validation rejects any path outside the run's own directory;
7. the push command is exactly `git push origin HEAD:main` with no force option;
8. aggregation refuses to emit summary files for an incomplete matrix and names
   the missing runs;
9. no code path writes to Google Drive, hashes a checkpoint, builds a ZIP
   archive, or calls `files.download`.
