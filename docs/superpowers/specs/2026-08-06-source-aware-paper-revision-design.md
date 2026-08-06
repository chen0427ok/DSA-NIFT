# Source-Aware Weight Evidence Paper Revision Design

**Date:** 2026-08-06  
**Target:** `/Users/brian/Rocling2026/paper/main.tex`

## Objective

Strengthen the rationale and evidence for the E19 source-aware arousal weights
without implying that the submitted configuration was chosen through a
hyperparameter search or that internal DSA-MST development results establish
target-domain generalization.

## Method Revision

The Source-Aware Supervision subsection will explain that the weights encode two
qualitative factors:

1. reflection-style similarity to the new-immigrant target texts; and
2. document granularity, especially whether labels describe isolated sentences
   or multi-sentence documents.

DSA-MST receives weight 1.0 as the reference, meaning that its arousal loss is
not down-weighted. It uses document-level, multi-sentence, first-person
reflections and the same 1--9 VA setup. CVAS receives 0.25 because its primarily
isolated general-domain sentences mismatch both style and granularity. CVAT
retains document granularity but remains general-domain, motivating the
intermediate 0.50. EDU2021 shares reflective style but is substantially shorter,
motivating the original 0.75.

The text will state explicitly that the quarter-step values are a simple ordinal
encoding of these priors, not estimates derived from document length, calibrated
domain similarity, or a parameter search.

## Sensitivity Analysis

Add a focused Results subsection and one compact table reporting three-seed
means and sample standard deviations on the fixed 253-document DSA-MST internal
development split. All four runs use the same MacBERT, L1++, batch size 32,
learning rate 2e-5, four epochs, maximum length 256, checkpoint criterion, and
seeds 42/1/2.

| Configuration | CVAS | CVAT | DSA-MST | EDU2021 | A-MAE | A-PCC |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 1.00 | 1.00 | 1.00 | 1.00 | .838±.014 | .606±.013 |
| Mild | .50 | .75 | 1.00 | 1.00 | .833±.015 | .609±.014 |
| Current | .25 | .50 | 1.00 | .75 | .830±.017 | .612±.016 |
| Reflection-swap | .25 | .50 | .75 | 1.00 | .830±.014 | .611±.015 |

The paragraph message is that the current configuration has the best mean
A-MAE and A-PCC in this small predeclared comparison, and every non-uniform
configuration improves mean arousal metrics over uniform weighting. However,
the changes are smaller than or comparable with between-seed sample standard
deviations. Current and reflection-swap are nearly indistinguishable, so the
evidence supports the coarse decision to down-weight general-domain arousal
supervision but does not identify the exact DSA-MST/EDU2021 ordering.

Valence metrics will not be added to this focused table because all valence
weights remain 1.0 and the reviewer question concerns arousal source weights.
The accompanying prose may note that shared parameters permit indirect valence
changes, but no valence claim will be based on this analysis.

## Historical and Evidential Framing

The original E19 configuration remains described as an a priori choice used for
the formal submission. The sensitivity analysis is explicitly post-hoc and is
not presented as the selection procedure that produced E19. It assesses whether
the heuristic behaves reasonably under nearby alternatives.

The analysis is internal-dev evidence only. The 12 test submission CSV files
contain predictions but no gold labels and therefore do not provide official
test MAE/PCC. No claim will label the sensitivity numbers as official validation
or official test results.

## Limitations Revision

Add one sentence noting that the sensitivity comparison uses a DSA-MST-derived
development split. Because this proxy shares the source favored by the current
scheme, it is not an independent target-domain validation and cannot establish
the globally optimal weights. Three seeds and four predeclared settings also do
not constitute a complete search.

## Verification

After editing:

1. compile `paper/main.tex` with the project's existing LaTeX workflow;
2. confirm there are no undefined references, multiply defined labels, or table
   overflow warnings attributable to the new material;
3. reverse-outline the revised Method and Results paragraphs;
4. verify every numerical claim against
   `sa_sensitivity_test_submissions/analysis/source_aware_summary.csv`;
5. adversarially check that no sentence implies official target-test evidence,
   statistical significance, or search-based selection.

