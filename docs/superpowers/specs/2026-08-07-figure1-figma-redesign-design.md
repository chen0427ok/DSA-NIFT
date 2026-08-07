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
- Use a compact two-column-paper aspect ratio of 1440 × 540 pixels rather than
  a 16:9 canvas.

## Information Architecture

The frame uses two horizontal lanes.

### Formal E19 submission

The upper lane occupies approximately 65–70% of the frame and is explicitly
divided into an inference path and training-only supervision. Both use solid
connectors, but their labels and colors remain distinct.

#### Inference path

The main left-to-right path shows:

1. Chinese reflection text;
2. MacBERT encoder;
3. masked mean pooling;
4. feature fusion;
5. a two-output regression head;
6. valence and arousal predictions.

A secondary input enters feature fusion from the lexicon branch:

`CVAW/CVAP + surface and migration cues → 31-d L1++ features → feature fusion`.

The L1++ box includes the secondary label `10 VA statistics + 21 surface/domain
cues`, making clear that L1++ is a handcrafted feature vector rather than a
lexicon embedding.

#### Training-only supervision

The training-only path shows the complete weighted-loss relationship:

- `Source identity → arousal source weights`;
- `Gold V/A labels → weighted Smooth L1 loss`;
- `Valence/arousal predictions → weighted Smooth L1 loss`;
- `Arousal source weights → weighted Smooth L1 loss`.

The source identity affects only the training loss and must not connect to the
inference representation. The formal lane must not contain pseudo-labeling,
synthetic examples, or graph-guided generation.

### Diagnostic alternatives

The lower lane occupies approximately 30–35% of the frame and uses pale fills
and dashed connectors. It contains three parallel alternatives rather than one
sequential pipeline:

1. `Multi-teacher pseudo-labeling`: two MacBERT teachers plus one RoBERTa
   teacher produce filtered labels.
2. `Graph-guided generation`: CVAW/CVAP plus FastText feed a kNN graph, seed
   selection, and LLM-generated reflections.
3. `Synthetic supervision alternatives`: direct labels, teacher labels, and
   pairwise constraints.

Graph-generated text connects by a dashed arrow to the synthetic-supervision
alternatives. Multi-teacher predictions connect by a dashed arrow specifically
to teacher labels. These relationships must not imply that pseudo-labeling,
graph generation, and pairwise constraints form one sequential method.

Official public-validation texts appear as an external reference box labeled
`style / length only` and `unlabeled; optional reference only`. A dashed arrow
connects this box only to LLM generation; it must not resemble an ordinary
training-data input.

The lane title explicitly says `Diagnostic alternatives — not used in formal
submission`.

## Visual Language

- White background and restrained academic-journal styling.
- Deep blue and solid connectors for formal inference.
- Violet and solid connectors for formal training-only supervision, lexicon
  features, and source-aware weighting.
- Pale lavender and dashed connectors for diagnostic branches.
- Slate gray for secondary labels and explanatory text.
- Encode lane semantics through text labels, color, and line style rather than
  color alone. Use the explicit headings `FORMAL E19 SUBMISSION` and
  `DIAGNOSTIC ALTERNATIVES — NOT USED IN FORMAL SUBMISSION`.
- Use rounded rectangles with 6–8 pixel corner radii, 1–1.25 pixel strokes,
  consistent 8-pixel spacing units, and no shadow. Do not add decorative
  illustration that competes with the method flow.
- Typography uses Inter with a clear hierarchy. After scaling to the paper's
  full text width, lane titles should remain approximately 8.5–9 pt, main-node
  text 7.5–8 pt, and secondary labels and legend text 6.5–7 pt. No text may
  fall below approximately 6.5 pt in the final paper.
- Use a three-chip legend in the upper-right corner: solid deep blue for formal
  inference, solid violet for formal training-only supervision, and dashed pale
  lavender for diagnostics. Omit a separate legend box.

## Paper Integration

Replace the inline `\fbox` construction with a full-width vector include while
preserving `\label{fig:pipeline}`. Use this caption:

> System overview. The upper lane shows the formal E19 system: MacBERT
> representations are fused with 31-dimensional L1++ features, while source
> identity affects only the arousal-weighted training loss. The lower lane shows
> diagnostic alternatives that were not used in the formal submission,
> including pseudo-labeling and graph-guided synthetic supervision.
> Public-validation texts are unlabeled and used only as optional style and
> length references.

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
- Confirm predictions, gold labels, and source weights all feed the weighted
  Smooth L1 loss.
- Confirm source identity has no connection to the inference representation.
- Confirm the three diagnostic methods are parallel alternatives rather than a
  false sequential pipeline.
- Confirm public-validation texts connect only to LLM style/length reference.
- Confirm the formal lane contains no pseudo-labeling or synthetic augmentation.
- Export as vector artwork and compile the paper.
- Verify text remains legible at full paper width, no text falls below the
  specified effective size, and the figure does not overflow the two-column
  layout.
- Inspect a grayscale rendering to confirm that headings and line styles retain
  the formal/training-only/diagnostic distinction without relying on color.
