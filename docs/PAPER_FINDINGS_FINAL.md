# Final Evidence-Based Paper Analysis

## 1. Executive summary

The strongest contribution is diagnostic: Valence is comparatively stable, while public-validation Arousal ordering did not transfer reliably to the larger official test set. Graph, ranking, and length controls changed their intended intermediate quantities, but these changes did not yield a clearly better Arousal calibration–ranking combination.

## 2. Task setting and evaluation caveat

Official validation contains 200 texts and official test 1,100 texts. Their gold labels are unavailable locally; all official scores are leaderboard returns. Internal DSA-MST development scores are not official validation scores.

## 3. Dimension-dependent reliability

For 6 paired primary systems, V-PCC validation/test Spearman rho is 0.123, while A-PCC rho is -0.574. Controlled ablation groups are reported in separate tables. With only 6 systems and p=0.234, this is descriptive evidence, not a significance claim.

## 4. Public-validation Arousal ranking did not transfer

E19 moved from 0.452 on public validation to 0.357 on test. RoBERTa-large moved from 0.407 to 0.390. We therefore frame public validation as an unreliable selector for Arousal among our submissions, not as universally invalid.

## 5. Valence is stable across systems

Valence PCC remains near 0.86–0.88 across paired systems and splits. This contrast makes the evaluation issue dimension-dependent rather than a blanket failure of the task.

## 6. L1 lexicon features

L1 hits 99.82% of official-test documents, yet its mean A-PCC differs from no-L1 by only 0.009. Safe claim: high coverage with limited measurable test utility under available seed evidence.

## 7. Synthetic augmentation and calibration–ranking trade-off

F2 has A-PCC 0.387 but A-MAE 1.037, versus 0.372 and 0.928 without augmentation. A small ranking increase co-occurs with worse absolute calibration.

## 8. Graph control and downstream performance

Graph expansion increases intrinsic seed coherence by 14.2× over VA lookup, but C→E changes mean test A-PCC by only about 0.003. The graph demonstrably changes the control signal; that intrinsic gain did not clearly transfer downstream.

## 9. Style and length alignment

F2 most strongly changes length alignment, with exact KS/Wasserstein values in `length_distribution.csv`. It still overproduces short texts and does not recover baseline A-MAE.

## 10. Recommended tables and figures

Use the generated val/test table, augmentation table, val-vs-test scatter, augmentation decomposition, and Arousal distribution plot. Treat prediction-distribution figures as label-free diagnostics.

## 11. Claims supported by evidence

- Validation Arousal ranking did not transfer reliably among paired submissions.
- Valence was more stable than Arousal.
- Graph expansion improved intrinsic seed coherence/diversity.
- F2 improved length-distribution proximity relative to fixed-length generation.
- L1 coverage is high, while test utility is not clearly measurable.

## 12. Claims not supported

- Source-aware loss, L1, graph augmentation, or pseudo-labels significantly improve official test performance.
- The graph is useless: intrinsic graph effects are measurable.
- Any per-document official prediction is correct; official gold labels are unavailable.

## 13. Suggested wording

**Abstract:** In a zero in-domain-label setting, we find stable Valence but non-transferring public-validation Arousal rankings. Controlled graph/LLM augmentation improves intrinsic control and prediction spread, yet does not yield a clearly better test calibration–ranking trade-off.

**Conclusion:** Our results motivate uncertainty-aware model selection and explicit separation of intrinsic controllability from downstream utility in dimensional sentiment shared tasks.
