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

    def write_test_csvs(self, root, submission_rows=None):
        test_input = root / "test.csv"
        submission = root / "submission.csv"
        test_input.write_text("ID,Text\na,甲\nb,乙\nc,丙\n", encoding="utf-8")
        rows = submission_rows or ["a,1,9", "b,5.5,4", "c,9,1"]
        submission.write_text("ID,Valence,Arousal\n" + "\n".join(rows) + "\n",
                              encoding="utf-8")
        return submission, test_input

    def test_test_command_uses_matching_checkpoint_and_official_input(self):
        cmd = self.ns["build_test_command"](
            "current", 42, pathlib.Path("model.pt"), pathlib.Path("out"))
        expected = {"--ckpt": "model.pt", "--input": "data/DSANIDF_TestSet.csv",
                    "--run_name": "sa_sensitivity_current_s42", "--split": "test",
                    "--lex_mode": "l1_intensity", "--out_dir": "out",
                    "--batch_size": "32", "--max_len": "256"}
        self.assertEqual(cmd[1], "predict.py")
        for flag, value in expected.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_twelve_test_submission_names_are_unique_and_exact(self):
        names = [self.ns["test_submission_name"](c, s)
                 for c, s in self.ns["expected_runs"]()]
        self.assertEqual(len(set(names)), 12)
        self.assertIn("sa_sensitivity_current_s42_test_submission.csv", names)

    def test_valid_test_submission_returns_hash_and_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            submission, test_input = self.write_test_csvs(pathlib.Path(tmp))
            result = self.ns["validate_test_submission"](
                submission, test_input, required_rows=3)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(len(result["sha256"]), 64)

    def test_submission_validation_rejects_wrong_id_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            submission, test_input = self.write_test_csvs(
                pathlib.Path(tmp), ["b,1,9", "a,5,4", "c,9,1"])
            with self.assertRaisesRegex(ValueError, "ID sequence"):
                self.ns["validate_test_submission"](submission, test_input, required_rows=3)

    def test_submission_validation_rejects_nonfinite_and_out_of_range(self):
        for bad, message in ((["a,nan,9", "b,5,4", "c,9,1"], "finite"),
                             (["a,0.9,9", "b,5,4", "c,9,1"], "range")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                submission, test_input = self.write_test_csvs(pathlib.Path(tmp), bad)
                with self.assertRaisesRegex(ValueError, message):
                    self.ns["validate_test_submission"](
                        submission, test_input, required_rows=3)

    def test_submission_validation_rejects_schema_count_and_bad_ids(self):
        cases = (
            ("ID,Valence,Wrong\na,1,9\nb,5,4\nc,9,1\n", "columns", 3),
            ("ID,Valence,Arousal\na,1,9\nb,5,4\n", "row count", 3),
            ("ID,Valence,Arousal\na,1,9\na,5,4\nc,9,1\n", "unique", 3),
            ("ID,Valence,Arousal\na,1,9\n,5,4\nc,9,1\n", "non-empty", 3),
        )
        for content, message, count in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                _, test_input = self.write_test_csvs(root)
                submission = root / "submission.csv"
                submission.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.ns["validate_test_submission"](
                        submission, test_input, required_rows=count)

    def test_receipt_contract_requires_test_artifacts(self):
        artifacts = self.ns["required_run_artifacts"]()
        self.assertIn("test_predictions.csv", artifacts)
        self.assertIn("test_submission.csv", artifacts)

    def test_publication_paths_cover_exact_matrix(self):
        pairs = self.ns["publication_paths"](pathlib.Path("drive"), pathlib.Path("repo"))
        self.assertEqual(len(pairs), 12)
        self.assertEqual(len({str(dst) for _, dst in pairs}), 12)
        self.assertEqual(str(pairs[0][1]),
                         "repo/test_submissions/sa_sensitivity_uniform_s42_test_submission.csv")

    def test_unexpected_staged_path_is_rejected(self):
        expected = [pathlib.Path("test_submissions/a.csv")]
        with self.assertRaisesRegex(RuntimeError, "unexpected staged paths"):
            self.ns["validate_staged_paths"](
                ["test_submissions/a.csv", "outputs/model.pt"], expected)

    def test_push_is_opt_in_non_force_and_token_is_not_embedded(self):
        self.assertEqual(self.ns["git_push_command"](),
                         ["git", "push", "origin", "HEAD:main"])
        self.assertNotIn("--force", self.source)
        self.assertNotIn("x-access-token:", self.source)
        self.assertIn("PUSH_TO_MAIN = False", self.source)
        self.assertIn("GIT_ASKPASS", self.source)


if __name__ == "__main__":
    unittest.main()
