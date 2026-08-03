# Controlled-ablation paper update design

## Goal

Replace the paper's batch-confounded lexicon and no-augmentation comparisons
with official-test results from a matched batch-32, three-seed experiment, then
synchronize the corresponding experiment documentation.

## Evidence

All four conditions use MacBERT, batch size 32, learning rate 2e-5, four epochs,
maximum length 256, the same checkpoint criterion, and seeds 42, 1, and 2.
Reported uncertainty is the sample standard deviation across those three seeds.

| Condition | V-MAE | V-PCC | A-MAE | A-PCC |
|---|---:|---:|---:|---:|
| No L1 | .617±.006 | .870±.000 | .923±.023 | .363±.021 |
| L1 | .613±.006 | .870±.000 | .923±.040 | .370±.000 |
| E | .630±.017 | .867±.006 | 1.057±.025 | .373±.012 |
| F2 | .630±.000 | .867±.006 | 1.073±.115 | .387±.012 |

The paper will describe two outcomes: L1 changes little relative to no L1, and
augmentation raises arousal PCC while worsening arousal MAE, with F2 showing the
largest PCC increase and the clearest PCC--MAE trade-off.

## Paper changes

- Add one focused matched-control table containing No L1, L1, E, and F2.
- Make the Abstract cite the matched L1/E/F2 comparison rather than the old
  batch-64 no-augmentation reference.
- Update Training Configuration to distinguish historical runs from the new
  matched batch-32 diagnostic without treating the latter as confounded.
- Rewrite Lexicon Comparison around the three-seed No-L1/L1 rows.
- Keep the historical component ladder for within-augmentation analysis, but
  use the new table for comparisons against no augmentation.
- Rewrite Discussion and Limitations to remove batch-confounded defenses.
- Do not retain sentences saying the differences are within seed variation or
  cannot be called significant. Also do not introduce an affirmative
  statistical-significance claim, because no inferential test was supplied.

## Documentation changes

- Synchronize `paper/tables/l1_ablation.tex` and
  `paper/tables/augmentation_ablation.tex` with the matched results.
- Add the four controlled conditions and per-seed values to
  `baseline/docs/RESULTS_SUMMARY.md` and `baseline/docs/test_results.md`.
- Mark the batch-controlled experiment complete in
  `baseline/docs/remaining_experiments.md`.
- Preserve historical archive files and the notebook design/implementation
  records because their batch-confounding text explains why the experiment was
  commissioned rather than stating the paper's current evidence.

## Verification

- Search active paper and experiment docs for obsolete batch-confounded phrases.
- Check every displayed mean and sample standard deviation against the 12 supplied
  leaderboard rows.
- Compile `paper/main.tex` and inspect the LaTeX log for errors, undefined
  references, and overfull boxes introduced by the new table.
- Reverse-outline the revised Abstract, Lexicon, Augmentation, Discussion, and
  Limitations paragraphs to ensure each makes one evidence-backed point.
