# Source-Aware Weight Sensitivity Colab Design

**Date:** 2026-08-04  
**Status:** Approved experiment concept; pending written-spec review

## Objective

Create a directly executable Google Colab notebook for a small sensitivity
analysis of the four arousal source weights used by E19. The experiment is a
predeclared robustness check, not a hyperparameter search: all configurations
are retained and reported, and no configuration is selected after inspecting
the results.

The analysis addresses four reviewer-facing questions:

1. Does source-aware weighting differ from uniform weighting?
2. Does a milder version preserve the same qualitative behavior?
3. Are the results dependent on the exact current values?
4. Is the ordering between DSA-MST and educational reflections important?

## Experimental Matrix

Weights below apply only to arousal. Valence remains weighted 1.0 for every
real source. The source order is CVAS, CVAT, DSA-MST, and EDU2021.

| Configuration | CVAS | CVAT | DSA-MST | EDU2021 | Purpose |
|---|---:|---:|---:|---:|---|
| `uniform` | 1.00 | 1.00 | 1.00 | 1.00 | No source-specific arousal down-weighting |
| `mild` | 0.50 | 0.75 | 1.00 | 1.00 | Test a gentler qualitative ordering |
| `current` | 0.25 | 0.50 | 1.00 | 0.75 | Reproduce the submitted E19 setting |
| `reflection_swap` | 0.25 | 0.50 | 0.75 | 1.00 | Test the DSA-MST versus EDU2021 ordering |

Each configuration is trained with seeds 42, 1, and 2, producing 12 matched
runs. The notebook must execute all 12 runs without requiring the user to edit
individual commands.

## Controlled Training Setup

All runs use the existing `train_v2.py` implementation and keep these settings
fixed:

- encoder: `hfl/chinese-macbert-base`;
- lexicon features: `l1_intensity` (31 dimensions);
- batch size: 32;
- epochs: 4;
- learning rate: 2e-5;
- maximum sequence length: 256;
- seeds: 42, 1, and 2;
- identical `data/train.csv` and DSA-MST-derived `data/dev.csv`;
- identical optimizer, scheduler, and checkpoint-selection criterion;
- no synthetic augmentation or ranking loss.

Every run passes all four source weights explicitly through
`--source_weights`; the notebook must not rely on implicit defaults. Run names
must encode both configuration and seed, for example
`sa_sensitivity_mild_s1`.

## Colab Workflow

The notebook will contain the following stages:

1. Verify that a GPU is available and print its model.
2. Clone the repository or reuse an existing checkout in Google Drive.
3. Install pinned or bounded Python dependencies compatible with the current
   training code.
4. Verify required training, development, lexicon, and external-resource files
   before launching expensive runs.
5. Define the four configurations and three seeds in one visible experiment
   table.
6. Run the 12 training commands sequentially, log stdout/stderr per run, and
   skip only runs whose expected checkpoint and development-prediction files
   already exist.
7. Parse every development prediction file independently, recompute V-MAE,
   V-PCC, A-MAE, and A-PCC, and save per-run results.
8. Aggregate mean, sample standard deviation, and paired-seed differences from
   `uniform` for every metric.
9. Export machine-readable CSV files, a publication-oriented LaTeX table, logs,
   predictions, and checkpoints into one ZIP archive.
10. Trigger a Colab download of the ZIP while also leaving it in Google Drive
    when Drive persistence is enabled.

## Outputs

The result directory will contain:

- `source_aware_per_run.csv`: one row per configuration and seed;
- `source_aware_summary.csv`: mean and sample SD for all four metrics;
- `source_aware_paired_vs_uniform.csv`: within-seed metric differences;
- `source_aware_summary.tex`: compact paper-ready table;
- `experiment_manifest.json`: configurations, commands, versions, timestamps,
  and file checksums where practical;
- `logs/*.log`: complete output from every run;
- `preds/*_dev.csv`: development predictions;
- `checkpoints/*_best.pt`: saved best checkpoints;
- `source_aware_sensitivity_results.zip`: downloadable bundle.

The notebook will display both the per-run and aggregated tables before
creating the archive. A failed run must remain visible in a status table and
must prevent the final summary from silently presenting an incomplete 12-run
experiment as complete.

## Analysis and Interpretation

Primary outcomes are DSA-MST development A-MAE and A-PCC. V-MAE and V-PCC are
secondary sanity checks because valence weights are unchanged, although shared
model parameters allow indirect differences.

The paper should report mean and sample SD across the three predeclared seeds,
plus paired-seed changes relative to `uniform`. The analysis should emphasize
patterns rather than select the numerically best configuration:

- similar `mild` and `current` results suggest that the qualitative ordering,
  rather than the exact values, drives behavior;
- similar `current` and `reflection_swap` results suggest that the strict
  DSA-MST/EDU2021 ordering is not important;
- materially different configurations indicate sensitivity to the heuristic
  choices and require a narrower claim;
- mixed or high-variance results do not establish a preferred weight vector.

DSA-MST weight 1.0 is a reference value meaning “not down-weighted,” not a
claim of perfect reliability. It is anchored by task format, document-level
granularity, Traditional Chinese, first-person reflective style, and the same
1–9 VA scale. EDU2021 shares reflective style but contains much shorter texts,
which motivates—but does not empirically prove—the original 0.75 choice.

Because the internal development split itself comes from DSA-MST, this analysis
cannot establish independent target-domain generalization or serve as a full
hyperparameter search. The limitation must be stated wherever the sensitivity
results are reported.

## Verification Criteria

The notebook is complete when:

1. all code cells execute in order in a clean Colab GPU runtime;
2. the generated commands cover exactly four configurations times three seeds;
3. each command explicitly fixes the full source-weight vector and controlled
   training arguments;
4. aggregation tests verify 12 unique configuration/seed rows and three seeds
   per configuration;
5. recomputed metrics match the metrics derived from each saved development
   prediction file within numerical tolerance;
6. missing or failed runs are reported explicitly;
7. the ZIP archive contains the manifest, tables, logs, predictions, and
   checkpoints needed for later analysis and reproducibility.

