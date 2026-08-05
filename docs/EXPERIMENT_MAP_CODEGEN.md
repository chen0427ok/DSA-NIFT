# Experiment Map (Code-Generated)

Generated from repository metadata and documented plans. `official_test_available=false` means no test claim is supported.

| ID | System | Category | Change | Val | Test | Supported finding |
|---|---|---|---|---:|---:|---|
| E1 | Baseline | baseline/data-source | EmoBank only | ✓ | — | official validation only |
| E2 | DAPT | baseline/data-source | MLM on 200 target-domain texts | ✓ | — | official validation only |
| E3 | Reflective corpora | baseline/data-source | Add DSA-MST and education reflection | ✓ | — | official validation only |
| E4 | L1 lexicon fusion | lexicon fusion | Add 10 CVAW/CVAP aggregates | ✓ | ✓ | official validation and test available; compare transfer rather than val-only gain |
| E5 | Raw synthetic augmentation | synthetic augmentation | 400 bin-centred LLM texts | ✓ | — | official validation only |
| E6 | E6 undocumented | undocumented | TODO | — | — | not run / no official evidence |
| E7 | E7 undocumented | undocumented | TODO | — | — | not run / no official evidence |
| E8 | E8 undocumented | undocumented | TODO | — | — | not run / no official evidence |
| E9 | E9 undocumented | undocumented | TODO | — | — | not run / no official evidence |
| E10 | 5-seed ensemble | ensemble | Mean of five MacBERT seeds | ✓ | — | official validation only |
| E11 | Encoder variants | encoder variant | RoBERTa base/large | ✓ | ✓ | official validation and test available; compare transfer rather than val-only gain |
| E12 | Multi-encoder ensemble | ensemble | Dimension-wise model fusion | ✓ | — | official validation only |
| E13 | Teacher pseudo-label | teacher pseudo-label | Teacher relabel 318 synthetic texts | ✓ | ✓ | official validation and test available; compare transfer rather than val-only gain |
| E14 | Frozen embedding + SVR | encoder variant | Frozen embedding regression | — | — | not run / no official evidence |
| E15 | Arousal/PCC loss weighting | ranking loss | Arousal and batch-PCC weights | — | — | not run / no official evidence |
| E16 | L1+L2 | lexicon fusion | OOV VA prediction features | — | — | not run / no official evidence |
| E17 | Arousal calibration | ablation | Post-hoc calibration | — | — | not run / no official evidence |
| E18 | L1++ intensity | lexicon fusion | 31 handcrafted features | ✓ | ✓ | official validation and test available; compare transfer rather than val-only gain |
| E19 | Source-aware loss | source-aware training | Down-weight general-corpus arousal | ✓ | ✓ | official validation and test available; compare transfer rather than val-only gain |
| E20 | Ranking-only augmentation | ranking loss | Synthetic pairwise hinge | ✓ | ✓ | official validation and test available; compare transfer rather than val-only gain |
| E21 | Dimension attention | attention/pooling | V/A-specific attention and multi-pooling | ✓ | — | official validation only |
| E22 | L1 + source-aware | ablation | Missing 2x2 cell; later planning reused ID | — | — | not run / no official evidence |
| E23 | Target-style RAG | synthetic augmentation | Unlabelled validation style anchors | — | — | not run / no official evidence |
| E24 | LLM enrichment features | lexicon fusion | Structured arousal indicators | — | — | not run / no official evidence |
| E25 | Frozen embedding stacking | encoder variant | Frozen embedding Ridge/SVR | — | — | not run / no official evidence |
| E26 | Arousal calibration | ablation | Post-processing calibration | — | — | not run / no official evidence |

## Interpretation discipline

Validation-only experiments answer what looked promising on the public validation set; they do not establish official-test utility. Planned or failed runs remain TODO rather than negative evidence.
