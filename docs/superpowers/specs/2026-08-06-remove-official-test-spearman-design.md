# Remove Official-Test Family Table and Spearman Narrative Design

**Date:** 2026-08-06  
**Target:** `/Users/brian/Rocling2026/paper/main.tex`

## Scope

Remove the six-family official-test comparison and every Spearman-dependent
claim while preserving:

- the formal E19 official-test score and fourth-place task result;
- the validation development trajectory;
- the matched official-test controlled ablation in `tab:controlled`;
- the bootstrap/seed diagnostics, source-weight sensitivity, and synthetic
  augmentation analysis.

## Evidence-Chain Removal

1. Convert `tab:main` from the combined validation/test table to a
   validation-only table with four metrics. Keep all 14 systems and the existing
   validation values. Update its caption, prose, and label to `tab:validation`.
2. Delete the entire `Method-Family Transfer to Test` subsection, including the
   six-family test comparison, both rank correlations, exact p-values, and the
   cross-split ordering interpretation.
3. Delete the Spearman calculation paragraph from Evaluation Protocol. Retain
   the formal/post-hoc distinction and the source-domain bootstrap protocol.
4. Remove Spearman and negative cross-split ordering claims from the Abstract,
   Introduction contributions, Related Work positioning, Task/Data terminology,
   and Conclusion.
5. Remove any residual references to `tab:main`, `sec:transfer`, six method
   families, negative rank association, or validation/test orderings.

## Replacement Narrative

Official public validation becomes the primary development narrative. The
validation-only main table will explain three decisions without making target-
test transfer claims: E4 gives the strongest balanced lexicon result, raw
synthetic supervision reaches the highest validation A-PCC (0.461) while
worsening A-MAE to 1.100, and E19 is submitted because it gives the lowest
validation A-MAE (0.870) together with high A-PCC (0.452). These values explain
the historical selection decision rather than establish final-test efficacy.

The E4--E18--E19 comparison will distinguish lexicon expansion from source
weighting. E18's 31-dimensional L1++ features alone underperform E4's 10 features
on all four validation metrics, so the paper will not claim that 31 dimensions
are independently superior. Holding the 31-dimensional representation fixed,
E19 source weighting improves arousal MAE from 0.909 to 0.870 and arousal PCC
from 0.408 to 0.452. The paper will therefore state that E19 was selected for its
validation arousal balance, not because lexicon expansion alone was beneficial.
Per the author's decision, this revision will not add a new limitation about the
unrun 10-dimensional plus source-aware factorial cell.

The Abstract will transition from the formal E19 score to the official-
validation PCC--MAE trade-off and then to controlled augmentation evidence. The
third contribution will focus on validation metric trade-offs and dimension-
dependent uncertainty supported by the 2.31-times wider arousal bootstrap
interval on the medical proxy and by the source-weight sensitivity analysis,
rather than by cross-split rank correlation.

Related Work will state only that the project studies finite-sample selection
uncertainty and metric trade-offs without claiming negatively associated
orderings. The Conclusion will summarize the supported evidence: official
validation exposes competing PCC/MAE preferences, arousal is more sample-
sensitive on the proxy, and the matched auxiliary-supervision runs yield a
PCC--MAE trade-off.

## Preserved Official-Test Evidence

`tab:controlled` and its dependent Lexicon/Synthetic Augmentation prose remain
unchanged because they are matched three-seed ablations rather than the removed
six-family/Spearman analysis. The formal E19 official-test score remains in the
Abstract and Evaluation Protocol as the prospective shared-task result.

## Verification

After editing:

1. `rg -i 'Spearman|tab:main|sec:transfer|six method|six-family|negatively
   associated|negative rank' paper/main.tex` returns no matches;
2. the validation-only table has four metric columns and no test columns;
3. all `\ref` and `\label` pairs resolve after compilation;
4. Tectonic/BibTeX compilation succeeds without undefined references or new
   overfull boxes;
5. the Abstract, Introduction, Results, and Conclusion contain no claim whose
   only evidence was the removed family table.
