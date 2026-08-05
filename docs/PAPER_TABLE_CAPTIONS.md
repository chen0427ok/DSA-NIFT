# Paper Table Captions

## `main_val_test_results.tex`
- Caption: Official validation and test results for systems evaluated on both splits.
- Interpretation: Valence is stable; Arousal ordering does not reliably transfer.
- Safe claim: Public validation was not a reliable Arousal selector among our paired submissions.
- Avoid: Validation is universally useless.

## `seed_variance.tex`
- Caption: Seed-level variation on official test leaderboard returns.
- Interpretation: Observed E4 seed variation is small at leaderboard precision.
- Safe claim: Training-seed variation does not explain the full val/test reversal.
- Avoid: True seed variance is exactly zero.

## `l1_ablation.tex`
- Caption: L1/no-L1 official test ablation.
- Interpretation: The observed mean difference is within available seed variation.
- Safe claim: L1 test utility is not clearly measurable here.
- Avoid: L1 is proven ineffective.

## `augmentation_ablation.tex`
- Caption: Controlled synthetic-augmentation component ladder.
- Interpretation: Arbitrary seed words and style/length alignment show larger observed changes than VA filtering or graph topology, while MAE remains worse than baseline.
- Safe claim: Intrinsic graph gains did not clearly transfer downstream.
- Avoid: The graph does nothing.

## `prediction_distribution.tex`
- Caption: Label-free prediction distribution on official test texts.
- Interpretation: Methods differ in mean/spread even without clearly different official scores.
- Safe claim: Distribution geometry changed.
- Avoid: Wider predictions are more correct.
