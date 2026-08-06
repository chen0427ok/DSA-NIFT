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

The Abstract will transition from the formal E19 score directly to controlled
augmentation evidence. The third contribution will focus on dimension-dependent
uncertainty supported by the 2.31-times wider arousal bootstrap interval on the
medical proxy and by the source-weight sensitivity analysis, rather than by
cross-split rank correlation.

Related Work will state only that the project studies finite-sample selection
uncertainty without claiming negatively associated orderings. The Conclusion
will summarize the supported uncertainty evidence: arousal is more sample-
sensitive on the proxy, and auxiliary supervision yields a PCC--MAE trade-off.

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

