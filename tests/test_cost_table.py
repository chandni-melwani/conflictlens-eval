import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "cost_table.py"))

SYSTEMS_FOLDERS = [
    ("conflictlens_structured_exact_test", True),
    ("conflictlens_structured_fuzzy_test", True),
    ("conflictlens_structured_none_test", True),
    ("conflictlens_hybrid_exact_test", True),
    ("nli_only_test", False),
    ("llm_judge_test", False),
    ("always_conflict_test", False),
]


def create_fixtures(base_dir: str, forbidden_text: str = None):
    gold_file = os.path.join(base_dir, "gold.jsonl")
    results_dir = os.path.join(base_dir, "results")
    expected_file = os.path.join(base_dir, "expected_cost.json")
    out_file = os.path.join(base_dir, "cost_table.txt")

    # Gold test items
    gold_items = [
        {"id": "t1", "split": "test", "label": 1},
        {"id": "t2", "split": "test", "label": 0},
    ]
    with open(gold_file, "w", encoding="utf-8") as f:
        for r in gold_items:
            f.write(json.dumps(r) + "\n")

    # Create preds and details for each system
    for folder, has_details in SYSTEMS_FOLDERS:
        sys_dir = os.path.join(results_dir, folder)
        os.makedirs(sys_dir, exist_ok=True)

        # Preds
        with open(os.path.join(sys_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "t1", "pred": 1, "score": 0.9}) + "\n")
            f.write(json.dumps({"id": "t2", "pred": 0, "score": 0.1}) + "\n")

        # Details if applicable
        if has_details:
            d1 = {
                "id": "t1",
                "no_pairs": False,
                "fallback_used": False,
                "n_candidate_pairs": 1,
                "p1_ok": True,
                "p2_ok": True,
            }
            d2 = {
                "id": "t2",
                "no_pairs": True,
                "fallback_used": False,
                "n_candidate_pairs": 0,
                "p1_ok": True,
                "p2_ok": True,
            }
            if forbidden_text:
                d1["passage_text"] = forbidden_text
                d1["fact_text"] = forbidden_text + "_fact"
                d2["passage_text"] = forbidden_text + "_2"

            with open(os.path.join(sys_dir, "details.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(d1) + "\n")
                f.write(json.dumps(d2) + "\n")

    expected_data = [
        {"system": "cl_structured_exact", "items": 2, "pairs_total": 1, "no_pair_items": 1, "nli_full_calls": 0},
        {"system": "cl_structured_fuzzy", "items": 2, "pairs_total": 1, "no_pair_items": 1, "nli_full_calls": 0},
        {"system": "cl_structured_none", "items": 2, "pairs_total": 1, "no_pair_items": 1, "nli_full_calls": 0},
        {"system": "cl_hybrid_exact", "items": 2, "pairs_total": 1, "no_pair_items": 1, "nli_full_calls": 0},
        {"system": "nli_only", "items": 2, "nli_full_calls": 2},
    ]
    with open(expected_file, "w", encoding="utf-8") as f:
        json.dump(expected_data, f, indent=2)

    return gold_file, results_dir, expected_file, out_file


class TestCostTable(unittest.TestCase):
    def test_match_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, results_dir, expected, out = create_fixtures(tmpdir)
            cmd = [
                sys.executable, SCRIPT_PATH,
                "--results-dir", results_dir,
                "--gold", gold,
                "--expected", expected,
                "--out", out,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Stderr: {res.stderr}")
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertTrue(content.strip().endswith("RESULT: ALL MATCH"))

    def test_changed_expected_pairs_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, results_dir, expected, out = create_fixtures(tmpdir)
            # Change pairs_total to mismatch
            with open(expected, "r", encoding="utf-8") as f:
                data = json.load(f)
            data[0]["pairs_total"] = 999
            with open(expected, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--results-dir", results_dir,
                "--gold", gold,
                "--expected", expected,
                "--out", out,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 1)
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("MISMATCH", content)
            self.assertTrue(content.strip().endswith("RESULT: 1 MISMATCHES"))

    def test_missing_details_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, results_dir, expected, out = create_fixtures(tmpdir)
            # Remove a details file
            os.remove(os.path.join(results_dir, "conflictlens_structured_exact_test", "details.jsonl"))

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--results-dir", results_dir,
                "--gold", gold,
                "--expected", expected,
                "--out", out,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 2)
            self.assertFalse(os.path.exists(out))
            self.assertIn("Missing input file", res.stderr)

    def test_passage_text_in_unread_field_not_in_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            forbidden = "SECRET_PASSAGE_SAMPLE_FORBIDDEN_TEXT_12345"
            gold, results_dir, expected, out = create_fixtures(tmpdir, forbidden_text=forbidden)

            cmd = [
                sys.executable, SCRIPT_PATH,
                "--results-dir", results_dir,
                "--gold", gold,
                "--expected", expected,
                "--out", out,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Stderr: {res.stderr}")
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn(forbidden, content)

    def test_two_runs_give_byte_identical_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold, results_dir, expected, out1 = create_fixtures(tmpdir)
            out2 = os.path.join(tmpdir, "cost_table_2.txt")

            cmd1 = [
                sys.executable, SCRIPT_PATH,
                "--results-dir", results_dir,
                "--gold", gold,
                "--expected", expected,
                "--out", out1,
            ]
            cmd2 = [
                sys.executable, SCRIPT_PATH,
                "--results-dir", results_dir,
                "--gold", gold,
                "--expected", expected,
                "--out", out2,
            ]
            res1 = subprocess.run(cmd1, capture_output=True, text=True)
            self.assertEqual(res1.returncode, 0, f"Stderr: {res1.stderr}")
            res2 = subprocess.run(cmd2, capture_output=True, text=True)
            self.assertEqual(res2.returncode, 0, f"Stderr: {res2.stderr}")

            with open(out1, "rb") as f1, open(out2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main()

