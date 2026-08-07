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

The L1++ box contains only the primary label `31-d L1++ feature vector` because
Table 2 documents the feature composition. Other secondary labels are removed
or shortened so that all remaining text stays legible after scaling to the
paper's full text width.

#### Training-only supervision

The training-only path shows the complete weighted-loss relationship:

- `Source identity → arousal source weights`;
- `Gold V/A labels → weighted Smooth L1 loss`;
- `Valence/arousal predictions → weighted Smooth L1 loss`;
- `Arousal source weights → weighted Smooth L1 loss`.

These three inputs enter the loss at visibly different vertical positions, so
their arrowheads and paths do not overlap.

The source identity affects only the training loss and must not connect to the
inference representation. The formal lane must not contain pseudo-labeling,
synthetic examples, or graph-guided generation.

### Diagnostic alternatives

The lower lane occupies approximately 30–35% of the frame and uses pale fills
and dashed connectors. It presents two diagnostic data sources that converge
on synthetic supervision:

1. `Multi-teacher pseudo-labeling` feeds `Synthetic supervision`.
2. `Graph-guided generation`: CVAW/CVAP plus FastText feed a kNN graph, seed
   selection, and LLM-generated reflections, which also feed `Synthetic
   supervision`.

Both branches connect independently to `Synthetic supervision`; they must not
appear to feed one another.

Official public-validation texts appear as a compact external reference box. A
dashed arrow labeled `style/length only` connects this box only to graph-guided
generation; it must not resemble an ordinary training-data input.

The lane title explicitly says `Diagnostic alternatives — not used in formal
submission`.

## Visual Language

- White background and restrained academic-journal styling.
- Deep blue, slightly heavier strokes, and solid connectors for formal
  inference. This is the strongest visual layer.
- A lighter violet and thinner solid connectors for formal training-only
  supervision, lexicon features, and source-aware weighting.
- Pale lavender fills and dashed connectors for diagnostic branches. This is
  the quietest visual layer.
- Slate gray for secondary labels and explanatory text.
- Encode lane semantics through text labels, color, and line style rather than
  color alone. Use the explicit headings `FORMAL E19 SUBMISSION` and
  `DIAGNOSTIC ALTERNATIVES — NOT USED IN FORMAL SUBMISSION`.
- Use rounded rectangles with 4–6 pixel corner radii, 1–1.5 pixel strokes,
  consistent 8-pixel spacing units, and no shadow. Do not add decorative
  illustration that competes with the method flow.
- Typography uses Inter with a clear hierarchy. Lane titles are smaller but
  heavier than before, while main-node labels are enlarged by removing
  nonessential secondary copy. No retained text may become marginal at the
  paper's full text width.
- Use a three-chip legend in the upper-right corner: solid deep blue for formal
  inference, solid violet for formal training-only supervision, and dashed pale
  lavender for diagnostics. Omit a separate legend box.

## Paper Integration

Replace the inline `\fbox` construction with a full-width vector include while
preserving `\label{fig:pipeline}`. Use this caption:

> System overview. The upper lane shows the formal E19 system, where MacBERT
> representations are fused with 31-dimensional L1++ features, while source
> identity affects only the arousal-weighted training loss. The lower lane shows
> diagnostic alternatives not used in the formal submission; public-validation
> texts are unlabeled and used only as optional style/length references.

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
  Smooth L1 loss through distinct, non-overlapping entry paths.
- Confirm source identity has no connection to the inference representation.
- Confirm multi-teacher pseudo-labeling and graph-guided generation independently
  feed synthetic supervision through an explicit data flow.
- Confirm public-validation texts connect only to LLM style/length reference.
- Confirm the formal lane contains no pseudo-labeling or synthetic augmentation.
- Export as vector artwork and compile the paper.
- Verify text remains legible at full paper width, no text falls below the
  specified effective size, and the figure does not overflow the two-column
  layout.
- Inspect a grayscale rendering to confirm that headings and line styles retain
  the formal/training-only/diagnostic distinction without relying on color.
