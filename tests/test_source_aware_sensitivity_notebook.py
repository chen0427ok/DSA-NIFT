"""Contract tests for the generated source-aware sensitivity Colab notebook.

Verification criteria: docs/superpowers/specs/2026-08-05-source-aware-sensitivity-rewrite-design.md
"""
import ast
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Rocling2026_Colab_source_aware_sensitivity.ipynb"
TEST_INPUT = ROOT / "data" / "DSANIDF_TestSet.csv"


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def load_namespace(nb):
    """Exec only the declarative nodes of the PURE_HELPERS cell."""
    source = next(s for s in code_cells(nb) if "# PURE_HELPERS" in s)
    tree = ast.parse(source)
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
               ast.FunctionDef, ast.ClassDef)
    module = ast.Module(body=[n for n in tree.body if isinstance(n, allowed)],
                        type_ignores=[])
    ns = {}
    exec(compile(module, "helpers", "exec"), ns)
    return ns


def write_submission(directory, ids, values=None, header="ID,Valence,Arousal"):
    path = pathlib.Path(directory) / "submission.csv"
    values = values or [(5.0, 5.0)] * len(ids)
    lines = [header] + [f"{i},{v},{a}" for i, (v, a) in zip(ids, values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class NotebookStructureTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.source = "\n".join("".join(c.get("source", [])) for c in self.nb["cells"])

    def test_ordinary_python_cells_compile(self):
        for index, source in enumerate(code_cells(self.nb)):
            if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                continue  # Colab magics are not valid Python
            with self.subTest(cell=index):
                compile(source, f"cell{index}", "exec")

    def test_notebook_is_deterministic(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "builder", ROOT / "notebooks" / "build_source_aware_sensitivity_notebook.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        rendered = json.dumps(builder.build_notebook(), ensure_ascii=False,
                              indent=1, sort_keys=True) + "\n"
        self.assertEqual(rendered, NOTEBOOK.read_text(encoding="utf-8"))

    def test_no_drive_zip_or_browser_download(self):
        """Defects 1-3 of the 2026-08-04 notebook must not reappear."""
        for banned in ("google.colab import drive", "drive.mount", "MyDrive",
                       "make_archive", "files.download", "import files"):
            self.assertNotIn(banned, self.source, f"{banned!r} must not reappear")

    def test_checkpoint_is_never_hashed_or_persisted(self):
        self.assertIn("checkpoint.unlink(missing_ok=True)", self.source)
        self.assertNotIn("checkpoint.pt", self.source)
        self.assertNotIn("sha256(checkpoint", self.source)

    def test_push_is_never_forced(self):
        self.assertNotIn("--force", self.source)
        self.assertNotIn("-f\"", self.source)
        self.assertNotIn("reset\", \"--hard", self.source)

    def test_main_loop_does_not_reraise(self):
        loop = next(s for s in code_cells(self.nb) if "for config, seed in expected_runs():" in s
                    and "statuses" in s)
        tree = ast.parse(loop)
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, ast.Raise)],
                         "a failed run must not abort the remaining runs")

    def test_no_token_literal_in_notebook(self):
        self.assertIn("getpass.getpass", self.source)
        self.assertNotIn("ghp_", self.source)
        self.assertNotIn("github_pat_", self.source)

    def test_push_access_is_verified_before_training(self):
        cells = code_cells(self.nb)
        probe = next(i for i, s in enumerate(cells) if "push\", \"--dry-run\"" in s)
        loop = next(i for i, s in enumerate(cells) if "statuses = []" in s)
        self.assertLess(probe, loop, "push permission must be checked before any training")


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_namespace(json.loads(NOTEBOOK.read_text(encoding="utf-8")))

    def test_exact_twelve_run_matrix(self):
        expected = [(c, s) for c in ("uniform", "mild", "current", "reflection_swap")
                    for s in (42, 1, 2)]
        self.assertEqual(self.ns["expected_runs"](), expected)

    def test_inverted_is_disabled_but_available(self):
        self.assertFalse(self.ns["ENABLE_INVERTED"])
        self.assertEqual(len(self.ns["build_configs"](False)), 4)
        enabled = self.ns["build_configs"](True)
        self.assertEqual(len(enabled) * len(self.ns["SEEDS"]), 15)
        self.assertEqual(enabled["inverted"],
                         {"sentence": 1.00, "text": 0.75, "reflection": 0.50, "edu2021": 0.25})

    def test_weight_specs_are_exact(self):
        self.assertEqual(self.ns["source_weight_spec"]("uniform"),
                         "sentence=1:1,text=1:1,reflection=1:1,edu2021=1:1")
        self.assertEqual(self.ns["source_weight_spec"]("mild"),
                         "sentence=1:0.5,text=1:0.75,reflection=1:1,edu2021=1:1")
        self.assertEqual(self.ns["source_weight_spec"]("current"),
                         "sentence=1:0.25,text=1:0.5,reflection=1:1,edu2021=1:0.75")
        self.assertEqual(self.ns["source_weight_spec"]("reflection_swap"),
                         "sentence=1:0.25,text=1:0.5,reflection=1:0.75,edu2021=1:1")

    def test_subprocesses_run_unbuffered(self):
        """Without -u the child buffers ~1.5 KB of stdout until exit and looks hung."""
        for command in (self.ns["build_train_command"]("current", 1),
                        self.ns["build_test_command"]("current", 1, "ck.pt", "out")):
            self.assertEqual(command[1], "-u", command)

    def test_train_command_fixes_every_controlled_argument(self):
        command = self.ns["build_train_command"]("current", 1)
        self.assertIn("train_v2.py", command)
        for flag, value in (("--model", "hfl/chinese-macbert-base"),
                            ("--lex_mode", "l1_intensity"),
                            ("--source_weights", "sentence=1:0.25,text=1:0.5,"
                                                 "reflection=1:1,edu2021=1:0.75"),
                            ("--run_name", "sa_sensitivity_current_s1"),
                            ("--seed", "1"), ("--epochs", "4"), ("--batch_size", "32"),
                            ("--lr", "2e-05"), ("--max_len", "256")):
            self.assertEqual(command[command.index(flag) + 1], value, flag)
        self.assertIn("--source_aware", command)
        for absent in ("--rank_aug", "--extra_train", "--pooling"):
            self.assertNotIn(absent, command)

    def test_test_command_targets_official_test_set(self):
        command = self.ns["build_test_command"]("mild", 42, "/content/ck.pt", "/content/out")
        self.assertIn("predict.py", command)
        for flag, value in (("--ckpt", "/content/ck.pt"),
                            ("--input", "data/DSANIDF_TestSet.csv"),
                            ("--run_name", "sa_sensitivity_mild_s42"),
                            ("--split", "test"), ("--lex_mode", "l1_intensity"),
                            ("--out_dir", "/content/out"), ("--max_len", "256")):
            self.assertEqual(command[command.index(flag) + 1], value, flag)

    def test_run_names_and_paths_are_unique(self):
        names = [self.ns["run_name"](c, s) for c, s in self.ns["expected_runs"]()]
        self.assertEqual(len(set(names)), 12)
        paths = [p for c, s in self.ns["expected_runs"]()
                 for p in self.ns["run_relative_paths"](c, s)]
        self.assertEqual(len(set(paths)), 48)
        self.assertTrue(all(p.startswith("results/source_aware_sensitivity/") for p in paths))

    def test_push_command_is_exact(self):
        self.assertEqual(self.ns["git_push_command"](), ["git", "push", "origin", "HEAD:main"])

    def test_commit_message_has_no_claude_attribution(self):
        message = self.ns["commit_message"]("current", 42)
        self.assertEqual(message, "sa-sensitivity: current s42 結果")
        self.assertNotIn("Claude", message)


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_namespace(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
        import pandas as pd
        self.ids = pd.read_csv(TEST_INPUT, dtype={"ID": str})["ID"].astype(str).tolist()
        self.assertEqual(len(self.ids), 1100)

    def _reject(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_submission(tmp, **kwargs)
            with self.assertRaises(ValueError):
                self.ns["validate_test_submission"](path, TEST_INPUT)

    def test_accepts_a_valid_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_submission(tmp, self.ids)
            result = self.ns["validate_test_submission"](path, TEST_INPUT)
            self.assertEqual(result["rows"], 1100)
            self.assertEqual(len(result["sha256"]), 64)

    def test_rejects_wrong_header(self):
        self._reject(ids=self.ids, header="ID,valence,arousal")

    def test_rejects_wrong_row_count(self):
        self._reject(ids=self.ids[:-1])

    def test_rejects_duplicate_ids(self):
        self._reject(ids=self.ids[:-1] + [self.ids[0]])

    def test_rejects_reordered_ids(self):
        self._reject(ids=list(reversed(self.ids)))

    def test_rejects_non_finite_values(self):
        self._reject(ids=self.ids, values=[("nan", 5.0)] * 1100)

    def test_rejects_out_of_range_values(self):
        self._reject(ids=self.ids, values=[(5.0, 9.5)] * 1100)
        self._reject(ids=self.ids, values=[(0.5, 5.0)] * 1100)


class StagingAndAggregationTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_namespace(json.loads(NOTEBOOK.read_text(encoding="utf-8")))

    def test_staging_rejects_paths_outside_the_run(self):
        expected = self.ns["run_relative_paths"]("current", 42)
        self.ns["validate_staged_paths"](expected, expected)          # exact
        self.ns["validate_staged_paths"](expected[:2], expected)      # subset is fine
        for intruder in ("data/train.csv", ".env",
                         "results/source_aware_sensitivity/sa_sensitivity_mild_s1/receipt.json",
                         "outputs/sa_sensitivity_current_s42_best.pt"):
            with self.assertRaises(RuntimeError):
                self.ns["validate_staged_paths"](expected + [intruder], expected)

    def _rows(self, drop=()):
        rows = []
        for i, (config, seed) in enumerate(self.ns["expected_runs"]()):
            if (config, seed) in drop:
                continue
            rows.append({"config": config, "seed": seed, "V_MAE": 0.60 + i * 0.001,
                         "V_PCC": 0.86, "A_MAE": 0.90 - i * 0.002, "A_PCC": 0.40})
        return rows

    def test_incomplete_matrix_is_named_and_refused(self):
        rows = self._rows(drop=[("mild", 1), ("current", 2)])
        self.assertEqual(self.ns["missing_runs"](rows), [("mild", 1), ("current", 2)])
        with self.assertRaises(RuntimeError) as ctx:
            self.ns["aggregate_rows"](rows)
        self.assertIn("mild", str(ctx.exception))

    def test_complete_matrix_aggregates(self):
        per_run, summary, paired = self.ns["aggregate_rows"](self._rows())
        self.assertEqual(len(per_run), 12)
        self.assertEqual(list(summary["config"]),
                         ["uniform", "mild", "current", "reflection_swap"])
        self.assertTrue((summary["n_seeds"] == 3).all())
        self.assertEqual(len(paired), 9)
        self.assertNotIn("uniform", set(paired["config"]))
        for metric in self.ns["METRICS"]:
            self.assertIn(f"{metric}_std", summary.columns)
            self.assertIn(f"delta_{metric}_vs_uniform", paired.columns)

    def test_paired_deltas_are_within_seed(self):
        rows = self._rows()
        _, _, paired = self.ns["aggregate_rows"](rows)
        lookup = {(r["config"], r["seed"]): r for r in rows}
        for _, row in paired.iterrows():
            expected = (lookup[(row["config"], row["seed"])]["A_MAE"]
                        - lookup[("uniform", row["seed"])]["A_MAE"])
            self.assertAlmostEqual(row["delta_A_MAE_vs_uniform"], expected, places=9)

    def test_latex_table_is_booktabs(self):
        _, summary, _ = self.ns["aggregate_rows"](self._rows())
        latex = self.ns["summary_latex"](summary)
        for token in (r"\toprule", r"\midrule", r"\bottomrule", r"reflection\_swap"):
            self.assertIn(token, latex)
        self.assertEqual(latex.count(r"\\"), 5)  # header + 4 configs


if __name__ == "__main__":
    unittest.main()
