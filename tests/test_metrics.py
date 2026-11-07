import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from metrics import (
    accuracy, auroc, balanced_accuracy, cluster_bootstrap, confusion,
    f1, f1_compatible, macro_f1, mcc, paired_bootstrap_diff, precision, recall, specificity,
    wilson_interval
)

try:
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestMetrics(unittest.TestCase):
    def test_fixture_1(self):
        # Fixture 1: TP=2, FP=1, FN=1, TN=2.
        # Expect precision, recall, specificity and F1 equal to 2/3, accuracy 2/3, balanced accuracy 2/3, MCC 1/3.
        labels = [1, 1, 0, 1, 0, 0]
        preds = [1, 1, 1, 0, 0, 0]
        tp, fp, fn, tn = confusion(labels, preds)
        self.assertEqual((tp, fp, fn, tn), (2, 1, 1, 2))

        self.assertAlmostEqual(precision(labels, preds), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(recall(labels, preds), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(specificity(labels, preds), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(f1(labels, preds), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(accuracy(labels, preds), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(balanced_accuracy(labels, preds), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(mcc(labels, preds), 1.0 / 3.0, places=7)

    def test_fixture_2(self):
        # Fixture 2 (always conflict, 3 positives and 1 negative):
        # TP=3, FP=1, FN=0, TN=0. Expect precision 0.75, recall 1.0, specificity 0.0, F1 6/7, balanced accuracy 0.5, MCC 0.0.
        labels = [1, 1, 1, 0]
        preds = [1, 1, 1, 1]
        tp, fp, fn, tn = confusion(labels, preds)
        self.assertEqual((tp, fp, fn, tn), (3, 1, 0, 0))

        self.assertAlmostEqual(precision(labels, preds), 0.75, places=7)
        self.assertAlmostEqual(recall(labels, preds), 1.0, places=7)
        self.assertAlmostEqual(specificity(labels, preds), 0.0, places=7)
        self.assertAlmostEqual(f1(labels, preds), 6.0 / 7.0, places=7)
        self.assertAlmostEqual(balanced_accuracy(labels, preds), 0.5, places=7)
        self.assertAlmostEqual(mcc(labels, preds), 0.0, places=7)

    def test_fixture_3(self):
        # Fixture 3: labels [0,0,1,1] and scores [0.1,0.4,0.35,0.8]. Expect AUROC 0.75.
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(auroc(labels, scores), 0.75, places=7)

    def test_fixture_4(self):
        # Fixture 4: all-positive labels. Expect specificity None, and that the bootstrap skips every resample without crashing.
        labels = [1, 1, 1, 1]
        preds = [1, 1, 0, 1]
        self.assertIsNone(specificity(labels, preds))

        items = [
            {"id": "1", "label": 1, "pred": 1, "article": "art1"},
            {"id": "2", "label": 1, "pred": 1, "article": "art2"},
            {"id": "3", "label": 1, "pred": 0, "article": "art3"},
        ]
        ci, skipped = cluster_bootstrap(
            items,
            lambda s: accuracy([x["label"] for x in s], [x["pred"] for x in s]),
            n=100,
            seed=42,
        )
        self.assertEqual(skipped, 100)
        self.assertEqual(ci, (None, None))

    def test_fixture_5(self):
        # Fixture 5: two runs of cluster_bootstrap with the same seed return identical intervals.
        # Resampling is by article, not by item.
        items = [
            {"id": "1", "label": 1, "pred": 1, "article": "A"},
            {"id": "2", "label": 1, "pred": 0, "article": "A"},
            {"id": "3", "label": 0, "pred": 0, "article": "B"},
            {"id": "4", "label": 0, "pred": 1, "article": "B"},
            {"id": "5", "label": 1, "pred": 1, "article": "C"},
            {"id": "6", "label": 0, "pred": 0, "article": "C"},
        ]

        ci1, skip1 = cluster_bootstrap(items, lambda s: mcc([x["label"] for x in s], [x["pred"] for x in s]), n=200, seed=42)
        ci2, skip2 = cluster_bootstrap(items, lambda s: mcc([x["label"] for x in s], [x["pred"] for x in s]), n=200, seed=42)
        self.assertEqual(ci1, ci2)
        self.assertEqual(skip1, skip2)

        # Verify resampling is by article, not by item:
        # Items 1 and 2 belong to article A, so their multiplicity in any resample must be identical.
        cluster_checked = []

        def verify_cluster_sampling(sample):
            c1 = sum(1 for x in sample if x["id"] == "1")
            c2 = sum(1 for x in sample if x["id"] == "2")
            cluster_checked.append(c1 == c2)
            return mcc([x["label"] for x in sample], [x["pred"] for x in sample])

        cluster_bootstrap(items, verify_cluster_sampling, n=100, seed=42)
        self.assertTrue(len(cluster_checked) > 0)
        self.assertTrue(all(cluster_checked))

    def test_fixture_6(self):
        # Fixture 6: evaluate.py stops with an error that lists the missing ids when a prediction is missing.
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_file = os.path.join(tmpdir, "gold.jsonl")
            preds_file = os.path.join(tmpdir, "preds.jsonl")

            with open(gold_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "item1", "split": "dev", "label": 1, "article": "a1"}) + "\n")
                f.write(json.dumps({"id": "item2", "split": "dev", "label": 0, "article": "a2"}) + "\n")
                f.write(json.dumps({"id": "item3", "split": "dev", "label": 1, "article": "a3"}) + "\n")

            with open(preds_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "item1", "pred": 1}) + "\n")

            eval_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluate.py"))
            proc = subprocess.run(
                [
                    sys.executable,
                    eval_script,
                    "--gold", gold_file,
                    "--preds", preds_file,
                    "--split", "dev",
                    "--name", "test_missing_fixture",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            combined_out = proc.stdout + proc.stderr
            self.assertIn("item2", combined_out)
            self.assertIn("item3", combined_out)

    def test_paired_bootstrap_diff(self):
        items = [
            {"id": "1", "label": 1, "pred_a": 1, "pred_b": 0, "article": "A"},
            {"id": "2", "label": 1, "pred_a": 1, "pred_b": 1, "article": "A"},
            {"id": "3", "label": 0, "pred_a": 0, "pred_b": 1, "article": "B"},
            {"id": "4", "label": 0, "pred_a": 0, "pred_b": 0, "article": "B"},
        ]
        diff, ci, skipped = paired_bootstrap_diff(
            items,
            lambda s: accuracy([x["label"] for x in s], [x["pred_a"] for x in s]),
            lambda s: accuracy([x["label"] for x in s], [x["pred_b"] for x in s]),
            n=100,
            seed=42,
        )
        self.assertIsNotNone(diff)
        self.assertAlmostEqual(diff, 1.0 - 0.5, places=5)
        self.assertIsNotNone(ci[0])
        self.assertIsNotNone(ci[1])

    def test_evaluate_cli_baseline_and_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_file = os.path.join(tmpdir, "gold.jsonl")
            preds_cmp = os.path.join(tmpdir, "preds_cmp.jsonl")
            with open(gold_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "1", "split": "dev", "label": 1, "article": "a1", "pair_tokens": 10, "contradictType": "numerical"}) + "\n")
                f.write(json.dumps({"id": "2", "split": "dev", "label": 0, "article": "a2", "pair_tokens": 50, "contradictType": "entity"}) + "\n")
                f.write(json.dumps({"id": "3", "split": "test", "label": 1, "article": "a3", "pair_tokens": 10}) + "\n")
            with open(preds_cmp, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "1", "pred": 0}) + "\n")
                f.write(json.dumps({"id": "2", "pred": 0}) + "\n")

            eval_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "evaluate.py"))
            proc = subprocess.run(
                [
                    sys.executable,
                    eval_script,
                    "--gold", gold_file,
                    "--baseline", "always_conflict",
                    "--split", "dev",
                    "--name", "test_baseline_run",
                    "--max_pair_tokens", "100",
                    "--group", "contradictType",
                    "--compare", preds_cmp,
                ],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Positives: 1, Negatives: 1", proc.stdout)
            self.assertIn("Recall by group on positives (contradictType):", proc.stdout)
            self.assertIn("Comparison vs", proc.stdout)
            self.assertIn("95% Wilson", proc.stdout)

            metrics_json_path = os.path.join(tmpdir, "results", "test_baseline_run", "metrics.json")
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            self.assertIn("precision_wilson_lo", res)
            self.assertIn("precision_wilson_hi", res)
            self.assertIn("recall_wilson_lo", res)
            self.assertIn("recall_wilson_hi", res)
            self.assertIn("specificity_wilson_lo", res)
            self.assertIn("specificity_wilson_hi", res)
            self.assertIn("precision_wilson_lo", res["metrics"]["Precision"])
            self.assertIn("recall_wilson_lo", res["metrics"]["Recall"])
            self.assertIn("specificity_wilson_lo", res["metrics"]["Specificity"])

    def test_wilson_interval(self):
        # wilson_interval(4, 4) -> approx (0.5101, 1.0), tolerance 1e-3
        lo, hi = wilson_interval(4, 4)
        self.assertAlmostEqual(lo, 0.5101, delta=1e-3)
        self.assertAlmostEqual(hi, 1.0, delta=1e-3)

        # wilson_interval(0, 51) -> approx (0.0, 0.0700), tolerance 1e-3
        lo, hi = wilson_interval(0, 51)
        self.assertAlmostEqual(lo, 0.0, delta=1e-3)
        self.assertAlmostEqual(hi, 0.0700, delta=1e-3)

        # wilson_interval(5, 10) -> approx (0.2366, 0.7634), tolerance 1e-3
        lo, hi = wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, delta=1e-3)
        self.assertAlmostEqual(hi, 0.7634, delta=1e-3)

        # wilson_interval(0, 0) -> (None, None)
        self.assertEqual(wilson_interval(0, 0), (None, None))

        # wilson_interval(3, 2) raises ValueError
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)

        # Negative k raises ValueError
        with self.assertRaises(ValueError):
            wilson_interval(-1, 5)

    def test_sklearn_checks(self):
        # If scikit-learn happens to be installed, add checks against its MCC and AUROC.
        # Skip those checks silently if it is not installed. Do not install it.
        if not HAS_SKLEARN:
            return
        labels = [1, 0, 1, 1, 0, 0, 1, 0]
        preds = [1, 0, 0, 1, 1, 0, 1, 0]
        scores = [0.9, 0.1, 0.4, 0.8, 0.7, 0.2, 0.85, 0.3]
        self.assertAlmostEqual(mcc(labels, preds), matthews_corrcoef(labels, preds), places=7)
        self.assertAlmostEqual(auroc(labels, scores), roc_auc_score(labels, scores), places=7)


if __name__ == "__main__":
    unittest.main()
