# Remove Official-Test Spearman Evidence Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the main result table to official-validation-only evidence, remove the six-family Spearman analysis, and rebuild the paper narrative around E4/E18/E19 validation decisions.

**Architecture:** Delete the family-transfer evidence chain from every section rather than leaving orphaned claims. Preserve formal E19 test reporting and the matched official-test controlled ablation, then compile and globally scan the paper for stale labels and language.

**Tech Stack:** LaTeX, ripgrep, Tectonic/BibTeX.

## Global Constraints

- Modify only `/Users/brian/Rocling2026/paper/main.tex` and generated LaTeX outputs.
- Keep the formal E19 test score/fourth-place statement.
- Keep `tab:controlled` and all matched augmentation results.
- Remove all Spearman values, methods, terminology, and cross-split ordering claims.
- Make `tab:main` validation-only and relabel it `tab:validation`.
- State that 31-dimensional L1++ alone underperforms 10-dimensional L1, while source-aware weighting improves arousal within the fixed 31-dimensional setting.
- Do not add the missing-factorial-cell limitation per author instruction.

---

### Task 1: Remove the evidence chain and rewrite the validation narrative

**Files:**
- Modify: `/Users/brian/Rocling2026/paper/main.tex`

**Interfaces:**
- Produces label `tab:validation`; removes `tab:main` and `sec:transfer`.

- [ ] **Step 1: Capture the pre-edit dependency scan**

Run `rg -n -i 'Spearman|tab:main|sec:transfer|six method|six-family|negatively associated|negative rank' paper/main.tex` and save the matched locations for deletion.

- [ ] **Step 2: Rewrite Abstract, Introduction, and Related Work**

Remove cross-split rank language. Replace it with official-validation metric
trade-offs and proxy uncertainty: raw synthetic has A-PCC .461/A-MAE 1.100;
E19 has A-PCC .452/A-MAE .870; the bootstrap arousal interval is 2.31 times
wider than valence on the medical proxy.

- [ ] **Step 3: Simplify Evaluation Protocol**

Delete family aggregation, rounding, permutation p-values, and paired-run caveats.
Retain formal-versus-post-hoc disclosure and the 5,000-resample proxy bootstrap.

- [ ] **Step 4: Convert the main table and Results opening**

Remove all four test columns and post-hoc caption text. Keep 14 validation rows.
Describe E4 as the strongest balanced lexicon result, E18 as evidence that L1++
alone does not help, and E19 as the arousal-oriented submission choice.

- [ ] **Step 5: Delete Method-Family Transfer and rewrite Conclusion**

Remove `sec:transfer` completely. Summarize official-validation MAE/PCC trade-offs,
proxy sampling sensitivity, source-weight sensitivity, and controlled augmentation
instead of six-family ordering.

### Task 2: Verify compilation and claim completeness

**Files:**
- Verify: `/Users/brian/Rocling2026/paper/main.tex`
- Verify: `/Users/brian/Rocling2026/paper/main.pdf`

**Interfaces:**
- Produces a compiled paper with no stale family-transfer evidence.

- [ ] **Step 1: Run the residual-language assertion**

Require no matches for `Spearman`, `tab:main`, `sec:transfer`, `six method`,
`six-family`, `negatively associated`, or `negative rank`.

- [ ] **Step 2: Assert preserved and replacement evidence**

Require the formal E19 test score, `tab:controlled`, `tab:validation`, and the
E4/E18/E19 validation deltas `.909`→`.870` and `.408`→`.452`.

- [ ] **Step 3: Compile and inspect the log**

Run `tectonic --keep-logs --keep-intermediates --reruns 2 main.tex`; reject
undefined references, multiply-defined labels, and new overfull boxes.

- [ ] **Step 4: Reverse-outline and adversarially review**

Check that Abstract/Introduction/Results/Conclusion each make claims supported by
remaining tables, without implying 31 dimensions are independently superior.

