import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "reconcile_counts.py"))

SYSTEMS_FOLDERS = [
    ("always_conflict", "always_conflict_dev", "always_conflict_test", None),
    ("always_compatible", "always_compatible_dev", "always_compatible_test", None),
    ("nli_only", "nli_only_dev", "nli_only_test", "nli_only_identical"),
    ("llm_judge", "llm_judge_dev", "llm_judge_test", "llm_judge_identical"),
    ("cl_structured_exact", "conflictlens_structured_exact_dev", "conflictlens_structured_exact_test", "conflictlens_structured_exact_identical"),
    ("cl_hybrid_exact", "conflictlens_hybrid_exact_dev", "conflictlens_hybrid_exact_test", "conflictlens_hybrid_exact_identical"),
    ("cl_structured_fuzzy", "conflictlens_structured_fuzzy_dev", "conflictlens_structured_fuzzy_test", None),
    ("cl_structured_none", "conflictlens_structured_none_dev", "conflictlens_structured_none_test", None),
]


def create_fixtures(base_dir: str):
    gold_file = os.path.join(base_dir, "gold.jsonl")
    gold_id_file = os.path.join(base_dir, "gold_identical.jsonl")
    results_dir = os.path.join(base_dir, "results")
    expected_file = os.path.join(base_dir, "expected.json")
    out_file = os.path.join(base_dir, "report.txt")

    # Gold rows
    dev_gold = [
        {"id": "d1", "split": "dev", "label": 1, "pair_tokens": 100},
        {"id": "d2", "split": "dev", "label": 0, "pair_tokens": 200},
    ]
    test_gold = [
        {"id": "t1", "split": "test", "label": 1, "pair_tokens": 100},
        {"id": "t2", "split": "test", "label": 0, "pair_tokens": 200},
    ]
    with open(gold_file, "w", encoding="utf-8") as f:
        for r in dev_gold + test_gold:
            f.write(json.dumps(r) + "\n")

    id_gold = [
        {"id": "i1", "split": "same_passage_analysis", "label": 1, "pair_tokens": 50}
    ]
    with open(gold_id_file, "w", encoding="utf-8") as f:
        for r in id_gold:
            f.write(json.dumps(r) + "\n")

    # Create preds folders and files
    for sys_name, dev_f, test_f, id_f in SYSTEMS_FOLDERS:
        dev_dir = os.path.join(results_dir, dev_f)
        os.makedirs(dev_dir, exist_ok=True)
        with open(os.path.join(dev_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "d1", "pred": 1, "score": 0.9}) + "\n")
            f.write(json.dumps({"id": "d2", "pred": 0, "score": 0.1}) + "\n")

        test_dir = os.path.join(results_dir, test_f)
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "t1", "pred": 1, "score": 0.9}) + "\n")
            f.write(json.dumps({"id": "t2", "pred": 0, "score": 0.1}) + "\n")

        if id_f:
            id_dir = os.path.join(results_dir, id_f)
            os.makedirs(id_dir, exist_ok=True)
            with open(os.path.join(id_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "i1", "pred": 0, "score": 0.0}) + "\n")

    # Expected JSON
    expected = [
        {"system": "nli_only", "split": "dev", "subset": "full", "n": 2, "tp": 1, "fp": 0, "fn": 0, "tn": 1, "mcc": 1.0},
        {"system": "nli_only", "split": "test", "subset": "full", "n": 2, "tp": 1, "fp": 0, "fn": 0, "tn": 1, "mcc": 1.0},
        {"system": "nli_only", "split": "identical", "subset": "full", "n": 1, "tp": 0, "fp": None, "fn": None, "tn": None, "mcc": None},
    ]
    with open(expected_file, "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)

    return gold_file, gold_id_file, results_dir, expected_file, out_file


class TestReconcileCounts(unittest.TestCase):
    def test_all_counts_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, gold_id, results_dir, expected, out = create_fixtures(tmpdir)
            cmd = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Stderr: {res.stderr}")
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue(content.strip().endswith("RESULT: ALL MATCH"))

    def test_one_expected_count_changed_reports_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, gold_id, results_dir, expected, out = create_fixtures(tmpdir)
            # Change one expected count
            with open(expected, "r", encoding="utf-8") as f:
                exp_data = json.load(f)
            exp_data[0]["tp"] = 99
            with open(expected, "w", encoding="utf-8") as f:
                json.dump(exp_data, f, indent=2)

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 1)
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("MISMATCH", content)
            self.assertTrue(content.strip().endswith("RESULT: 1 MISMATCHES"))

    def test_preds_file_missing_one_gold_id_reports_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, gold_id, results_dir, expected, out = create_fixtures(tmpdir)
            # Remove d2 from nli_only_dev
            preds_file = os.path.join(results_dir, "nli_only_dev", "preds.jsonl")
            with open(preds_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "d1", "pred": 1, "score": 0.9}) + "\n")

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 1)
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("missing 1", content)

    def test_missing_preds_file_exits_code_2_no_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, gold_id, results_dir, expected, out = create_fixtures(tmpdir)
            # Delete nli_only_dev preds.jsonl
            preds_file = os.path.join(results_dir, "nli_only_dev", "preds.jsonl")
            os.remove(preds_file)

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 2)
            self.assertFalse(os.path.exists(out))

    def test_running_twice_gives_byte_identical_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, gold_id, results_dir, expected, out1 = create_fixtures(tmpdir)
            out2 = os.path.join(tmpdir, "report2.txt")

            cmd1 = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out1
            ]
            res1 = subprocess.run(cmd1, capture_output=True, text=True)
            self.assertEqual(res1.returncode, 0)

            cmd2 = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out2
            ]
            res2 = subprocess.run(cmd2, capture_output=True, text=True)
            self.assertEqual(res2.returncode, 0)

            with open(out1, "rb") as f1, open(out2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_always_conflict_synthesis_line_when_no_preds_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, gold_id, results_dir, expected, out = create_fixtures(tmpdir)
            # Remove always_conflict dev and test preds.jsonl
            for split_f in ["always_conflict_dev", "always_conflict_test"]:
                p = os.path.join(results_dir, split_f, "preds.jsonl")
                if os.path.exists(p):
                    os.remove(p)

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--gold", gold,
                "--gold-identical", gold_id,
                "--results-dir", results_dir,
                "--expected", expected,
                "--out", out
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Stderr: {res.stderr}")
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            expected_line = "- always_conflict: no preds.jsonl in results; constant predictions (1) generated over gold ids of each split"
            self.assertIn(expected_line, content)


if __name__ == "__main__":
    unittest.main()

