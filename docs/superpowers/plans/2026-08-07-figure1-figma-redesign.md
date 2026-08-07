# Figure 1 Figma Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished, editable Figure 1 in the supplied Figma file, export it as SVG/PDF, and replace the inline LaTeX pipeline diagram.

**Architecture:** Create one 1440 × 540 top-level Figma frame with a formal upper lane and a diagnostic lower lane. Build and validate the frame incrementally through the Figma Plugin API, then export the finished frame and integrate the PDF asset into `paper/main.tex`.

**Tech Stack:** Figma Plugin API via `use_figma`, Figma asset export, SVG/PDF, LaTeX/Tectonic.

## Global Constraints

- Target Figma file key: `aVCMGwhi58wg4BJ1ugQOvx`; page `0:1`.
- Use Inter, white background, no shadow, 4–6 px radii, and restrained strokes.
- Formal inference is solid deep blue with the strongest stroke weight; formal training is lighter violet; diagnostics are pale lavender with dashed connectors.
- The final figure must remain legible at full paper text width and in grayscale.
- Formal E19 must contain no pseudo-labeling or synthetic augmentation.
- Public-validation text is unlabeled and connects only as an optional style/length reference.

---

### Task 1: Build and validate the editable Figma frame

**Files:**
- Modify remotely: Figma file `aVCMGwhi58wg4BJ1ugQOvx`, page `0:1`

**Interfaces:**
- Consumes: approved information architecture in `docs/superpowers/specs/2026-08-07-figure1-figma-redesign-design.md`.
- Produces: top-level frame `Figure 1 — System Pipeline` and its node ID.

- [ ] **Step 1: Inspect the blank page and available Inter font styles**

Run a read-only `use_figma` script returning page children and matching Inter font styles. Expected: page `0:1` and no existing top-level artwork.

- [ ] **Step 2: Create the 1440 × 540 frame and lane skeleton**

Create a white top-level frame, formal and diagnostic lane backgrounds, headings, dividers, and the three-chip legend. Return every created node ID and capture a screenshot.

- [ ] **Step 3: Build the formal inference path**

Create the six-node left-to-right inference path and the L1++ side input. Use solid deep-blue connectors for the main path and lighter violet for the lexicon branch. Keep only core labels and remove the L1++ composition subtitle documented in Table 2.

- [ ] **Step 4: Build formal training-only supervision**

Create source identity, arousal source weights, gold V/A labels, predictions, and weighted Smooth L1 loss. Connect all three required inputs to the loss at distinct vertical entry positions with light-violet lines; do not connect source identity to inference features.

- [ ] **Step 5: Build parallel diagnostic alternatives**

Create multi-teacher, graph-guided generation, and synthetic-supervision cards. Connect the first two independently to synthetic supervision with dashed arrows. Add the optional unlabeled public-validation reference only to graph generation, with `style/length only` on the dashed connector.

- [ ] **Step 6: Visually validate and refine**

Capture a high-resolution screenshot. Check for cropped text, overlaps, ambiguous arrow direction, minimum effective typography, and grayscale distinguishability. Apply only targeted fixes, then capture a final screenshot.

### Task 2: Export vector assets

**Files:**
- Create: `paper/figures/system_pipeline_figma.svg`
- Create: `paper/figures/system_pipeline_figma.pdf`

**Interfaces:**
- Consumes: final top-level Figma frame node ID from Task 1.
- Produces: SVG and PDF renderings of the same frame.

- [ ] **Step 1: Export the final frame as SVG and PDF**

Use the Figma asset export tool twice with the final frame node ID and explicit `svg` and `pdf` formats. Download the returned assets immediately to the exact paths above.

- [ ] **Step 2: Validate exported files**

Run:

```bash
file paper/figures/system_pipeline_figma.svg paper/figures/system_pipeline_figma.pdf
test -s paper/figures/system_pipeline_figma.svg
test -s paper/figures/system_pipeline_figma.pdf
```

Expected: non-empty SVG and PDF files with the reported vector formats.

### Task 3: Integrate Figure 1 into the paper

**Files:**
- Modify: `paper/main.tex`
- Regenerate: `paper/main.pdf`

**Interfaces:**
- Consumes: `paper/figures/system_pipeline_figma.pdf`.
- Produces: compiled paper using the new Figure 1 while preserving `fig:pipeline`.

- [ ] **Step 1: Replace the inline box diagram**

Replace the existing `figure*` body with:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/system_pipeline_figma.pdf}
\caption{System overview. The upper lane shows the formal E19 system, where MacBERT
representations are fused with 31-dimensional L1++ features, while source
identity affects only the arousal-weighted training loss. The lower lane shows
diagnostic alternatives not used in the formal submission; public-validation
texts are unlabeled and used only as optional style/length references.}
\label{fig:pipeline}
\end{figure*}
```

- [ ] **Step 2: Compile and inspect diagnostics**

Run:

```bash
cd paper
tectonic --keep-logs --keep-intermediates --reruns 2 main.tex
rg -i 'undefined references?|multiply defined|LaTeX Error|Emergency stop|Overfull' main.log
```

Expected: Tectonic exits 0; the diagnostic search returns no matches introduced by Figure 1.

- [ ] **Step 3: Verify replacement scope**

Run:

```bash
rg -n 'system_pipeline_figma|fig:pipeline|\\fbox' paper/main.tex
```

Expected: one figure asset reference, two occurrences of `fig:pipeline` (reference and label), and no `\fbox` construction in Figure 1.
