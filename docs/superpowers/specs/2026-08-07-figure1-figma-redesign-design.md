# Figure 1 Figma Redesign

## Objective

Replace the inline LaTeX box diagram in `paper/main.tex` with a polished,
editable Figma vector figure that makes the formal E19 submission visually
distinct from diagnostic alternatives. The figure must remain legible at the
two-column width used by the paper.

## Figma Target

- File: `https://www.figma.com/design/aVCMGwhi58wg4BJ1ugQOvx/Untitled`
- Page: `Page 1` (`0:1`)
- The page is currently empty.
- Create one top-level landscape frame named `Figure 1 — System Pipeline`.

## Information Architecture

The frame uses two horizontal lanes.

### Formal E19 submission

The upper lane occupies approximately 70% of the frame and uses solid
connectors. It shows:

1. source-labeled Chinese reflections;
2. MacBERT encoder;
3. masked mean pooling;
4. fusion with the 31-dimensional L1++ affective-lexicon feature vector;
5. a two-output regression head producing valence and arousal;
6. source identity mapped to arousal weights and then to the weighted Smooth L1
   training loss.

This lane must not imply that E19 uses synthetic examples.

### Diagnostic alternatives

The lower lane occupies approximately 30% of the frame and uses pale fills and
dashed connectors. It contains:

- multi-teacher pseudo-labeling;
- CVAW/CVAP plus FastText, kNN graph seeds, and LLM-generated reflections;
- direct labels, teacher labels, and pairwise constraints as alternative forms
  of synthetic supervision;
- optional public-validation style and length references.

The lane title explicitly says `Diagnostic alternatives — not used in formal
submission`.

## Visual Language

- White background and restrained academic-journal styling.
- Deep blue for the formal inference/training path.
- Violet for lexicon and source-aware supervision.
- Pale lavender and dashed strokes for diagnostic branches.
- Slate gray for secondary labels and explanatory text.
- Rounded rectangles with minimal shadow, consistent 8-pixel spacing units,
  and no decorative illustration that competes with the method flow.
- Typography uses Inter with a clear hierarchy and remains readable after the
  figure is scaled to the paper's full text width.
- A compact legend distinguishes formal path, training-only supervision, and
  diagnostic alternatives.

## Deliverables

1. Editable Figma frame in the supplied file.
2. Exported vector PDF and SVG in `paper/figures/`.
3. `paper/main.tex` updated to use the exported Figure 1 asset while preserving
   the existing `fig:pipeline` label and an evidence-accurate caption.
4. A compiled `paper/main.pdf` with no undefined references or overfull boxes
   introduced by the replacement.

## Validation

- Inspect the completed Figma frame via screenshot at high resolution.
- Confirm every formal and diagnostic connection against the Methodology text.
- Confirm the formal lane contains no pseudo-labeling or synthetic augmentation.
- Export as vector artwork and compile the paper.
- Verify text remains legible and the figure does not overflow the two-column
  layout.
