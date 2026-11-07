import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.paired_checks import (
    find_saved_scores_file,
    paired_auroc_bootstrap,
    run_check_a,
)


class TestPairedChecks(unittest.TestCase):
    def setUp(self):
        self.labels = [1, 0, 1, 0, 1, 0, 1, 0]
        self.scores_a = [0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.75, 0.25]
        self.scores_b = [0.6, 0.4, 0.5, 0.5, 0.55, 0.45, 0.7, 0.3]

    def test_identical_systems_give_diff_zero_and_ci_contains_zero_auroc(self):
        val_a, val_b, diff, ci, skipped = paired_auroc_bootstrap(
            self.labels, self.scores_a, self.scores_a, n=200, seed=42
        )
        self.assertEqual(diff, 0.0)
        self.assertIsNotNone(ci[0])
        self.assertIsNotNone(ci[1])
        self.assertLessEqual(ci[0], 0.0)
        self.assertGreaterEqual(ci[1], 0.0)

    def test_swapped_sign_case_flips_sign_auroc(self):
        val_a, val_b, diff_ab, ci_ab, _ = paired_auroc_bootstrap(
            self.labels, self.scores_a, self.scores_b, n=200, seed=42
        )
        val_b2, val_a2, diff_ba, ci_ba, _ = paired_auroc_bootstrap(
            self.labels, self.scores_b, self.scores_a, n=200, seed=42
        )
        self.assertAlmostEqual(diff_ba, -diff_ab, places=6)
        self.assertAlmostEqual(ci_ba[0], -ci_ab[1], places=6)
        self.assertAlmostEqual(ci_ba[1], -ci_ab[0], places=6)

    def test_identical_systems_give_diff_zero_and_ci_contains_zero_check_a(self):
        mock_items = [
            {"id": f"item_{i}", "label": self.labels[i], "article": f"art_{i % 3}"}
            for i in range(len(self.labels))
        ]
        map_a = {f"item_{i}": {"id": f"item_{i}", "pred": 1 if self.scores_a[i] >= 0.5 else 0} for i in range(len(self.labels))}
        res = run_check_a(mock_items, map_a, map_a, n=200, seed=42)
        self.assertEqual(res["diff_mcc"], 0.0)
        self.assertLessEqual(res["ci_mcc"][0], 0.0)
        self.assertGreaterEqual(res["ci_mcc"][1], 0.0)
        self.assertEqual(res["diff_bacc"], 0.0)
        self.assertLessEqual(res["ci_bacc"][0], 0.0)
        self.assertGreaterEqual(res["ci_bacc"][1], 0.0)

    def test_swapped_sign_case_flips_sign_check_a(self):
        mock_items = [
            {"id": f"item_{i}", "label": self.labels[i], "article": f"art_{i % 3}"}
            for i in range(len(self.labels))
        ]
        map_a = {f"item_{i}": {"id": f"item_{i}", "pred": 1 if self.scores_a[i] >= 0.5 else 0} for i in range(len(self.labels))}
        map_b = {f"item_{i}": {"id": f"item_{i}", "pred": 1 if self.scores_b[i] >= 0.5 else 0} for i in range(len(self.labels))}

        res_ab = run_check_a(mock_items, map_a, map_b, n=200, seed=42)
        res_ba = run_check_a(mock_items, map_b, map_a, n=200, seed=42)

        self.assertAlmostEqual(res_ba["diff_mcc"], -res_ab["diff_mcc"], places=6)
        self.assertAlmostEqual(res_ba["diff_bacc"], -res_ab["diff_bacc"], places=6)
        self.assertAlmostEqual(res_ba["ci_mcc"][0], -res_ab["ci_mcc"][1], places=6)
        self.assertAlmostEqual(res_ba["ci_mcc"][1], -res_ab["ci_mcc"][0], places=6)

    def test_missing_saved_scores_file_detection(self):
        missing = find_saved_scores_file(["non_existent_file_123.jsonl", "another_fake.jsonl"])
        self.assertIsNone(missing)


if __name__ == "__main__":
    unittest.main()

