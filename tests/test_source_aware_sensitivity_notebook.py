import ast
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Rocling2026_Colab_source_aware_sensitivity.ipynb"


def load_namespace(nb):
    source = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "# PURE_HELPERS" in "".join(c["source"])
    )
    tree = ast.parse(source)
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
               ast.FunctionDef, ast.ClassDef)
    module = ast.Module(body=[n for n in tree.body if isinstance(n, allowed)], type_ignores=[])
    ns = {}
    exec(compile(module, "helpers", "exec"), ns)
    return ns


class NotebookContractTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.source = "\n".join("".join(c.get("source", [])) for c in self.nb["cells"])
        self.ns = load_namespace(self.nb)

    def test_exact_twelve_run_matrix(self):
        expected = [(c, s) for c in ("uniform", "mild", "current", "reflection_swap")
                    for s in (42, 1, 2)]
        self.assertEqual(self.ns["expected_runs"](), expected)

    def test_current_and_swap_weight_specs_are_exact(self):
        self.assertEqual(
            self.ns["source_weight_spec"]("current"),
            "sentence=1:0.25,text=1:0.5,reflection=1:1,edu2021=1:0.75",
        )
        self.assertEqual(
            self.ns["source_weight_spec"]("reflection_swap"),
            "sentence=1:0.25,text=1:0.5,reflection=1:0.75,edu2021=1:1",
        )

    def test_training_command_is_fully_controlled(self):
        cmd = self.ns["build_train_command"]("mild", 2)
        expected_pairs = {
            "--seed": "2", "--batch_size": "32", "--epochs": "4",
            "--lr": "2e-05", "--max_len": "256", "--lex_mode": "l1_intensity",
        }
        for flag, value in expected_pairs.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn("--source_aware", cmd)
        self.assertNotIn("--rank_aug", cmd)
        self.assertNotIn("--extra_train", cmd)

    def test_dev_metrics_are_computed_from_saved_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dev.csv"
            path.write_text(
                "valence_true,arousal_true,valence_pred,arousal_pred\n"
                "1,1,1,1\n2,2,3,2\n3,3,3,2\n", encoding="utf-8")
            got = self.ns["compute_dev_metrics"](path)
        self.assertAlmostEqual(got["V_MAE"], 1 / 3)
        self.assertAlmostEqual(got["A_MAE"], 1 / 3)
        self.assertAlmostEqual(got["V_PCC"], 0.8660254038)
        self.assertAlmostEqual(got["A_PCC"], 0.8660254038)

    def test_incomplete_matrix_is_rejected(self):
        rows = []
        for config, seed in self.ns["expected_runs"]()[:-1]:
            rows.append({"config": config, "seed": seed, "V_MAE": 1.0,
                         "V_PCC": 0.5, "A_MAE": 1.0, "A_PCC": 0.4})
        with self.assertRaisesRegex(RuntimeError, "incomplete 12-run matrix"):
            self.ns["aggregate_rows"](rows)

    def test_complete_matrix_has_sample_sd_and_nine_paired_rows(self):
        rows = []
        for config, seed in self.ns["expected_runs"]():
            offset = {42: 0.0, 1: 1.0, 2: 2.0}[seed]
            rows.append({"config": config, "seed": seed, "V_MAE": 1 + offset,
                         "V_PCC": .5, "A_MAE": 2 + offset, "A_PCC": .4})
        per_run, summary, paired = self.ns["aggregate_rows"](rows)
        self.assertEqual(len(per_run), 12)
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(paired), 9)
        self.assertAlmostEqual(summary.loc[summary.config == "uniform", "V_MAE_std"].iloc[0], 1.0)

    def test_notebook_has_safe_resume_and_download_contract(self):
        for fragment in ("drive.mount", "validate_inputs", "receipt.json", "os.replace",
                         "sha256", "source_aware_summary.tex", "files.download",
                         "incomplete 12-run matrix"):
            self.assertIn(fragment, self.source)


if __name__ == "__main__":
    unittest.main()
