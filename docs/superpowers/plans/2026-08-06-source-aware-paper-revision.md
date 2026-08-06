# Source-Aware Weight Paper Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise `paper/main.tex` so the source-aware weights have a concrete qualitative rationale and a correctly bounded three-seed internal-development sensitivity analysis.

**Architecture:** Update the existing Method subsection with the prior behind the ordinal weights, add one focused Results subsection/table immediately after uncertainty diagnostics, and extend Limitations with the proxy-dev/search boundary. Verify all numbers from the generated CSV and compile the complete paper before claiming completion.

**Tech Stack:** LaTeX, booktabs tables, Python/CSV numerical checks, XeLaTeX/BibTeX.

## Global Constraints

- Preserve the historical fact that E19 weights were chosen a priori before formal submission.
- Do not describe sensitivity metrics as official validation or official test results.
- Do not claim statistical significance or globally optimal weights.
- Report exactly four configurations over seeds 42/1/2 on the fixed 253-document DSA-MST internal development split.
- Report arousal MAE/PCC as mean ± sample SD to three decimals.
- State that current is best in mean A-MAE/A-PCC but current and reflection-swap are nearly indistinguishable.
- Explain `1.0` as an undown-weighted reference and quarter steps as ordinal encoding, not calibrated similarity.
- Modify only `/Users/brian/Rocling2026/paper/main.tex`; preserve unrelated baseline worktree changes.

---

### Task 1: Verify the numerical evidence

**Files:**
- Read: `sa_sensitivity_test_submissions/analysis/source_aware_per_run.csv`
- Read: `sa_sensitivity_test_submissions/analysis/source_aware_summary.csv`
- Read: `sa_sensitivity_test_submissions/analysis/source_aware_paired_vs_uniform.csv`

**Interfaces:**
- Produces the exact table values consumed by Task 2.

- [ ] **Step 1: Recompute group means and sample standard deviations**

Run a Python check that groups per-run data by `config`, uses `std(ddof=1)`, and
asserts equality with the summary CSV within `1e-12`.

- [ ] **Step 2: Verify directional claims**

Assert that every non-uniform row has lower mean A-MAE and higher mean A-PCC
than uniform, current has the best mean values, and absolute current/swap gaps
are below `.002` for both arousal metrics.

### Task 2: Revise Method, Results, and Limitations

**Files:**
- Modify: `/Users/brian/Rocling2026/paper/main.tex`

**Interfaces:**
- Produces label `tab:source-sensitivity` and subsection label `sec:source-sensitivity`.

- [ ] **Step 1: Revise Source-Aware Supervision**

Replace the short weight-rationale sentences with one paragraph whose first
sentence states the two qualitative factors, then defines DSA-MST as reference,
explains CVAS/CVAT/EDU2021, and states that quarter steps are neither length-
derived nor similarity-calibrated.

- [ ] **Step 2: Add the sensitivity subsection and table**

Insert after Sampling and Seed Diagnostics. Use columns Configuration, CVAS,
CVAT, DSA-MST, EDU2021, A-MAE↓, A-PCC↑ and these rows:

```text
Uniform         1.00 1.00 1.00 1.00  .838±.014 .606±.013
Mild             .50  .75 1.00 1.00  .833±.015 .609±.014
Current          .25  .50 1.00  .75  .830±.017 .612±.016
Reflection-swap  .25  .50  .75 1.00  .830±.014 .611±.015
```

The discussion must distinguish consistent mean direction from uncertainty and
state that the swap comparison does not identify the exact reflective-source
ordering.

- [ ] **Step 3: Extend Limitations**

Add a sentence explaining that DSA-MST-derived dev favors the same proxy family
used by the weighting prior, and four settings × three seeds are not a complete
search or independent target-domain validation.

### Task 3: Compile and adversarially review

**Files:**
- Verify: `/Users/brian/Rocling2026/paper/main.tex`
- Verify: `/Users/brian/Rocling2026/paper/main.pdf`

**Interfaces:**
- Produces a compiled paper with resolved references and evidence-calibrated prose.

- [ ] **Step 1: Compile the full paper**

From `paper/`, run XeLaTeX, BibTeX, then XeLaTeX twice. Require exit code 0.

- [ ] **Step 2: Audit the log**

Search `main.log` for undefined control sequences, undefined references,
multiply-defined labels, and overfull boxes attributable to the new table.

- [ ] **Step 3: Reverse-outline revised paragraphs**

Confirm Method says rationale → mapping → non-search caveat; Results says protocol
→ numbers → bounded interpretation; Limitations says proxy bias → no optimum claim.

- [ ] **Step 4: Run claim-evidence assertions**

Search the revised subsection for forbidden implications (`significant`,
`optimal`, official `test`/`validation` attribution) and manually verify that
every number matches Task 1.

- [ ] **Step 5: Report the edit**

Provide clickable links to `main.tex` and `main.pdf`, the compiled page count,
and a compact claim-evidence map.

