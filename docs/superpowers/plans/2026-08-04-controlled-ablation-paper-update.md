# Controlled-ablation Paper Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace batch-confounded paper comparisons with the supplied matched batch-32, three-seed official-test results and synchronize active experiment documentation.

**Architecture:** Treat the 12 leaderboard rows as the sole source for the new controlled table, calculate mean and sample standard deviation once, then copy the verified values into the paper and active docs. Preserve historical component-ladder evidence but route every no-augmentation and lexicon conclusion through the new matched table.

**Tech Stack:** LaTeX, Markdown, Python/pandas for numeric verification, `latexmk` or `pdflatex` for paper compilation.

## Global Constraints

- Conditions use seeds 42, 1, and 2 with batch size 32, learning rate 2e-5, four epochs, maximum length 256, and the same checkpoint criterion.
- Report sample standard deviation (`ddof=1`) across the three seeds.
- Remove active-paper wording that calls lexicon or no-augmentation comparisons batch-confounded or not pure ablations.
- Do not retain statements that the controlled differences fall within seed variation or cannot be called significant.
- Do not add an affirmative statistical-significance claim because no inferential test was supplied.
- Preserve historical archive, notebook design, and implementation records.
- Preserve unrelated dirty-worktree changes; stage only files intentionally edited by this plan.

---

### Task 1: Record and verify the controlled results in active docs

**Files:**
- Modify: `docs/RESULTS_SUMMARY.md`
- Modify: `docs/test_results.md`
- Modify: `docs/remaining_experiments.md`

**Interfaces:**
- Consumes: the 12 supplied official-test rows.
- Produces: one canonical per-seed table and one mean±sample-SD table used by the paper edit.

- [ ] **Step 1: Run the numeric source-of-truth calculation**

Use pandas with the exact rows below and `groupby(...).agg(["mean", "std"])`:

```python
rows = [
    ("E",42,.62,.87,1.08,.38), ("E",1,.62,.87,1.03,.36), ("E",2,.65,.86,1.06,.38),
    ("F2",42,.63,.87,1.19,.38), ("F2",1,.63,.86,.96,.38), ("F2",2,.63,.87,1.07,.40),
    ("L1",42,.61,.87,.90,.37), ("L1",1,.61,.87,.90,.37), ("L1",2,.62,.87,.97,.37),
    ("No L1",42,.61,.87,.95,.38), ("No L1",1,.62,.87,.91,.34), ("No L1",2,.62,.87,.91,.37),
]
```

Expected rounded aggregates:

```text
No L1  .617±.006  .870±.000  .923±.023  .363±.021
L1     .613±.006  .870±.000  .923±.040  .370±.000
E      .630±.017  .867±.006  1.057±.025  .373±.012
F2     .630±.000  .867±.006  1.073±.115  .387±.012
```

- [ ] **Step 2: Add the controlled experiment to active docs**

Add the 12 raw rows, the aggregate table, the matched hyperparameters, and these
descriptive deltas:

```text
L1 − No L1: V-MAE −.003, V-PCC .000, A-MAE .000, A-PCC +.007.
E − L1: V-MAE +.017, V-PCC −.003, A-MAE +.133, A-PCC +.003.
F2 − L1: V-MAE +.017, V-PCC −.003, A-MAE +.150, A-PCC +.017.
```

Mark the batch-controlled experiment complete in `remaining_experiments.md`.

- [ ] **Step 3: Verify doc values and scope**

Run a Python assertion that all four aggregate strings appear in both result
documents, then run:

```bash
git diff --check -- docs/RESULTS_SUMMARY.md docs/test_results.md docs/remaining_experiments.md
```

Expected: assertions pass and `git diff --check` reports no errors.

- [ ] **Step 4: Commit active-doc results**

```bash
git add docs/RESULTS_SUMMARY.md docs/test_results.md docs/remaining_experiments.md
git commit -m "記錄 batch-32 controlled ablation 官方結果"
```

### Task 2: Replace the paper's confounded comparisons

**Files:**
- Modify: `../paper/main.tex`
- Modify: `../paper/tables/l1_ablation.tex`
- Modify: `../paper/tables/augmentation_ablation.tex`

**Interfaces:**
- Consumes: the verified aggregates from Task 1.
- Produces: one matched-control table and consistent Abstract, Setup, Results, Discussion, and Limitations prose.

- [ ] **Step 1: Establish the pre-edit failure signal**

Run:

```bash
rg -n -i "batch-confounded|contrast is confounded|not a pure|L1 uses batch 64|no-augmentation reference uses batch 64" ../paper/main.tex
```

Expected: matches in the Abstract, Training Configuration, Lexicon Comparison,
augmentation caption/prose, Discussion, and Limitations.

- [ ] **Step 2: Add the matched-control table**

Insert a four-row table with columns System, Seeds, Batch, V-MAE, V-PCC, A-MAE,
and A-PCC. Use exactly three decimals for means and SDs, arrows in metric headers,
and a caption stating that encoder, seeds, batch size, learning rate, epochs,
maximum length, and checkpoint selection are matched.

- [ ] **Step 3: Rewrite paper claims**

Make these section-level messages explicit:

```text
Abstract: F2 changes A-PCC .370→.387 while A-MAE changes .923→1.073 under matched settings.
Setup: historical runs had exceptions, and the new four-condition diagnostic reruns the relevant contrasts at batch 32 with seeds 42/1/2.
Lexicon: L1 changes A-PCC .363→.370 while mean A-MAE remains .923.
Augmentation: E gives .373/1.057 A-PCC/A-MAE and F2 gives .387/1.073, versus .370/.923 without augmentation.
Discussion: the matched comparison confirms a PCC--MAE trade-off rather than an optimization-setting artifact.
Limitations: remove batch confounding; retain other limitations such as target-gold access, post-hoc evaluation, F style/length coupling, one LLM family, and synthetic-label quality.
```

Do not include the prohibited seed-variation/significance caveat.

- [ ] **Step 4: Synchronize standalone table files**

Update `l1_ablation.tex` to the matched No-L1/L1 comparison. Update
`augmentation_ablation.tex` so its no-augmentation comparison uses the matched
L1 row and its controlled E/F2 values match the new results.

- [ ] **Step 5: Reverse-outline the revised prose**

For each revised paragraph, check that its first sentence conveys the section
message and every numeric sentence supports that message. Remove duplicate
defenses and do not mix the historical batch-64 seed diagnostic into the new
causal comparison.

### Task 3: Verify paper consistency and compilation

**Files:**
- Verify: `../paper/main.tex`
- Verify: `../paper/main.pdf`
- Verify: `../paper/main.log`
- Verify: active docs and standalone tables from Tasks 1–2.

**Interfaces:**
- Consumes: all updated prose and tables.
- Produces: compiled paper and an evidence-backed completion report.

- [ ] **Step 1: Search for obsolete active-paper defenses**

Run:

```bash
rg -n -i "batch-confounded|contrast is confounded|not a pure|L1 uses batch 64|no-augmentation reference uses batch 64|within.*seed|cannot.*significant" ../paper/main.tex
```

Expected: no matches.

- [ ] **Step 2: Check controlled values programmatically**

Assert that `.617`, `.613`, `.630`, `.923`, `1.057`, `1.073`, `.363`, `.370`,
`.373`, and `.387` occur in the matched table and that its caption states batch
32 and three seeds.

- [ ] **Step 3: Compile the paper**

Run from `../paper`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable, run `pdflatex`, `bibtex`, then `pdflatex` twice.
Expected: exit code 0 and an updated `main.pdf`.

- [ ] **Step 4: Audit the LaTeX log and diffs**

Run:

```bash
rg -n "Undefined control sequence|LaTeX Error|undefined references|Overfull \\hbox" ../paper/main.log
git diff --check
```

Expected: no LaTeX errors or undefined references; no newly introduced overfull
box in the matched table; no whitespace errors.

- [ ] **Step 5: Commit any final baseline-doc corrections**

The `paper/` directory is outside the `baseline` Git repository and therefore
cannot be included in its commit. If verification changes active baseline docs,
stage only those files and commit with:

```bash
git commit -m "修正 controlled ablation 文件一致性"
```
