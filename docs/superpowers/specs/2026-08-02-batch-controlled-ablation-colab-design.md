# Batch-controlled ablation Colab notebook design

## Goal

Create a focused Google Colab notebook that removes batch-size confounding from
the paper's lexicon and augmentation comparisons. The notebook must train the
following four MacBERT conditions with seeds 42, 1, and 2:

| Condition | Lexicon | Extra training data | Batch size |
|---|---|---|---:|
| `nolex_b32` | none | none | 32 |
| `l1_b32` | L1 | none | 32 |
| `aug_E_b32` | L1 | `data/train_aug_E.csv` | 32 |
| `aug_F2_b32` | L1 | `data/train_aug_F2.csv` | 32 |

The `l1_b32` condition is shared by the lexicon and augmentation ablations, so
the full experiment requires 12 rather than 15 training runs.

## Reproducibility contract

Every run uses the same:

- public GitHub repository and recorded Git commit;
- resolved `hfl/chinese-macbert-base` Hugging Face revision;
- seed set: 42, 1, and 2;
- batch size: 32;
- learning rate: 2e-5;
- maximum sequence length: 256;
- training duration: four epochs;
- optimizer, warmup, weight decay, dropout, gradient clipping, and checkpoint
  criterion implemented by `train_v2.py`.

The notebook passes the important settings explicitly rather than relying on
training-script defaults. A preflight cell verifies the required scripts and
CSV files before starting any GPU work. The run manifest records the repository
commit, encoder revision, package versions, GPU model, condition, seed, and
training arguments.

## Notebook structure

The notebook is a new file rather than an extension of the earlier KG ablation
notebook. Its cells are organized as follows:

1. Explain the two controlled comparisons and expected outputs.
2. Install dependencies and verify a CUDA GPU.
3. Clone or refresh the public repository's `main` branch.
4. Mount Google Drive and create a persistent result directory.
5. Resolve and cache one immutable MacBERT snapshot for all runs.
6. Define and display the four-condition experiment matrix.
7. Run preflight validation, including data columns and row counts.
8. Train each condition/seed, immediately run official validation and test
   inference, calculate internal-development metrics, and persist small
   artifacts.
9. Resume safely by consulting per-run completion receipts on Drive.
10. Summarize per-seed and per-condition internal-development results.
11. Build leaderboard-ready zip files and an empty official-score entry sheet.
12. Export one compact handoff archive for paper revision.

## Execution and persistence

Each run has a unique name such as `controlled_l1_b32_s42`. Training is invoked
through `subprocess` so the exact argument list can be logged. A run is marked
complete only after all required small artifacts have been copied to Drive and
a completion receipt has been written atomically.

For each successful run, the notebook:

1. retains the training log and effective configuration;
2. copies internal-development predictions and metrics;
3. copies the organizer-validation predictions/submission;
4. invokes `predict.py` on `data/DSANIDF_TestSet.csv` and copies its predictions;
5. creates validation and test submission zip files whose internal filename is
   `submission.csv`;
6. records the artifact filenames in the receipt;
7. optionally deletes the large checkpoint after all inference and persistence
   checks succeed.

On a fresh Colab runtime, receipts allow completed runs to be skipped without
retaining all 12 checkpoints. A force-rerun switch remains available for an
explicit restart.

## Outputs

The final handoff archive contains:

- `run_manifest.csv` and a JSON environment manifest;
- `dev_metrics_per_seed.csv`;
- `dev_metrics_summary.csv` with mean and sample standard deviation;
- internal-development and organizer-validation prediction CSV files;
- 12 validation submission zip files;
- 12 test submission zip files;
- `official_scores_to_fill.csv`, with one row per run and empty V-MAE, V-PCC,
  A-MAE, and A-PCC fields for the user to fill after leaderboard evaluation;
- the per-run training logs and completion receipts.

The paper will not be edited in this task. After the user returns official
scores, a separate revision will update tables, claims, and batch-confounding
language based on the controlled results.

## Failure handling

- Missing GPU, scripts, model snapshot, or data files stops execution before
  training.
- A failed training or prediction command does not create a completion receipt.
- Existing artifacts without a valid receipt are treated as incomplete and may
  be regenerated.
- Checkpoints are deleted only after all required inference files exist and the
  Drive copy has been verified.
- The final summary reports failed and incomplete runs separately; it never
  silently computes a three-seed mean from fewer than three seeds.

## Verification

Before handoff, validate the notebook JSON, compile every code cell as Python
after excluding notebook shell/magic lines, and run a no-training static smoke
test of the experiment matrix, filenames, manifest generation, summary logic,
and submission zip layout. Full GPU training remains a Colab user action.
