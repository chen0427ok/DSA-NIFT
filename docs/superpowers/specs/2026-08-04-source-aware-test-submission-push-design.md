# Source-Aware Test Submission Push Design

**Date:** 2026-08-04  
**Status:** Approved concept; pending written-spec review

## Objective

Extend `Rocling2026_Colab_source_aware_sensitivity.ipynb` so that all 12 trained
models produce official-format test predictions. After every artifact passes
validation, the notebook creates one commit containing exactly the 12 test
submission CSV files and pushes that commit to `origin/main`.

## Test Inference Artifacts

Immediately after a run's training checkpoint is available, invoke `predict.py`
with:

- the run's exact checkpoint;
- `--lex_mode l1_intensity`;
- `--input data/DSANIDF_TestSet.csv`;
- the same run name;
- `--split test`;
- a run-local temporary output directory.

Each Drive run directory retains `test_predictions.csv` and
`test_submission.csv` in addition to the existing checkpoint, development and
validation predictions, log, and receipt. Their file sizes and SHA-256 hashes
are part of the atomic completion receipt, so an older receipt without test
artifacts is considered incomplete and reruns test inference before it becomes
eligible for final publication.

## Validation Contract

Every `test_submission.csv` must satisfy all of the following:

- header is exactly `ID,Valence,Arousal`;
- it contains exactly 1,100 data rows;
- IDs are non-empty and unique;
- its ID sequence exactly matches `data/DSANIDF_TestSet.csv`;
- Valence and Arousal are finite numeric values in the inclusive range 1–9;
- the file is non-empty and has a recorded SHA-256 digest.

No Git commit or push is attempted until the exact 4-config × 3-seed matrix has
12 valid receipts and 12 valid test submissions.

## Repository Layout

The final commit contains only these generated files:

```text
test_submissions/sa_sensitivity_uniform_s42_test_submission.csv
test_submissions/sa_sensitivity_uniform_s1_test_submission.csv
test_submissions/sa_sensitivity_uniform_s2_test_submission.csv
test_submissions/sa_sensitivity_mild_s42_test_submission.csv
test_submissions/sa_sensitivity_mild_s1_test_submission.csv
test_submissions/sa_sensitivity_mild_s2_test_submission.csv
test_submissions/sa_sensitivity_current_s42_test_submission.csv
test_submissions/sa_sensitivity_current_s1_test_submission.csv
test_submissions/sa_sensitivity_current_s2_test_submission.csv
test_submissions/sa_sensitivity_reflection_swap_s42_test_submission.csv
test_submissions/sa_sensitivity_reflection_swap_s1_test_submission.csv
test_submissions/sa_sensitivity_reflection_swap_s2_test_submission.csv
```

The commit message is `加入 source-aware sensitivity test predictions`.
Checkpoints, predictions with diagnostic columns, logs, receipts, summaries,
tokens, and temporary authentication helpers must not be staged.

## Safe Git and Authentication Flow

The token is collected once through a hidden Colab prompt and retained only in
memory. Clone and push authenticate through a temporary `GIT_ASKPASS` helper
with restrictive permissions. The helper is removed in a `finally` block. The
token must never appear in the stored origin URL, command output, exception
message, manifest, receipt, or notebook source.

Before creating the commit, the notebook:

1. fetches `origin/main`;
2. requires the current checkout to fast-forward cleanly to `origin/main`;
3. copies the 12 validated Drive submissions to `test_submissions/`;
4. stages exactly those 12 explicit paths;
5. rejects unexpected staged paths;
6. creates one commit only when the staged files differ from `HEAD`;
7. pushes with `git push origin HEAD:main` and never force-pushes.

Git author name and email are copied from the latest repository commit. If the
remote advances between fetch and push, the non-force push is allowed to fail;
the notebook reports that the local commit is preserved and requires the user
to rerun the publication cell after reviewing the new remote state.

If all 12 remote files already have the same content, rerunning the publication
cell reports completion without creating a duplicate commit. If any same-name
remote file differs or the checkout cannot fast-forward, the cell stops rather
than overwriting or rewriting history.

## Notebook Flow and Failure Handling

Test inference is part of `run_one`, so training/test artifacts remain aligned
and resumable. The final publication cell is deliberately separate from the
training and result-download cells. It displays the 12 source and destination
paths, validates them again, asks for an explicit in-cell confirmation value
(`PUSH_TO_MAIN = True`), and only then performs the commit and push.

Training or test-inference failure creates no valid receipt. Validation failure,
unexpected staged content, Git divergence, authentication failure, or push
rejection stops the publication cell with a clear error and does not use force,
reset, or destructive cleanup.

## Verification Criteria

The change is complete when automated tests demonstrate that:

1. `predict.py` commands reference the correct checkpoint, test input, run name,
   split, output directory, and lexicon mode;
2. all 12 destination names are deterministic and unique;
3. malformed columns, row counts, IDs, non-finite values, and out-of-range
   predictions are rejected;
4. receipts without test artifacts are not accepted;
5. publication rejects incomplete matrices and unexpected staged paths;
6. authentication helpers are temporary and the notebook contains no embedded
   token;
7. the push target is exactly `origin HEAD:main` without a force option;
8. the generated notebook remains deterministic JSON and all ordinary Python
   cells compile.

