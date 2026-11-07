import json
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.metrics import wilson_interval


class TestScoreExtractionAudit(unittest.TestCase):
    def test_score_extraction_audit_counts_and_wilson_interval_on_fake_inputs(self):
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "score_extraction_audit.py")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = os.path.join(tmpdir, "fake_labels.jsonl")
            key_path = os.path.join(tmpdir, "fake_key.jsonl")
            output_path = os.path.join(tmpdir, "fake_results.txt")

            fake_labels = [
                {
                    "audit_id": "aud_01",
                    "facts": [
                        {"S": "y", "V": "y", "A": "y"},
                        {"S": "y", "V": "y", "A": "y"},
                    ],
                    "C": "y",
                },
                {
                    "audit_id": "aud_02",
                    "facts": [
                        {"S": "y", "V": "n", "A": "y"},
                    ],
                    "C": "y",
                },
                {
                    "audit_id": "aud_03",
                    "facts": [
                        {"S": "n", "V": None, "A": None},
                    ],
                    "C": "n",
                },
                {
                    "audit_id": "aud_04",
                    "facts": [
                        {"S": "y", "V": "y", "A": "n"},
                    ],
                    "C": "y",
                },
            ]

            fake_key = [
                {"audit_id": "aud_01", "item_id": "item_pos_1", "passage_index": 1, "gold_label": 1},
                {"audit_id": "aud_02", "item_id": "item_pos_2", "passage_index": 2, "gold_label": 1},
                {"audit_id": "aud_03", "item_id": "item_neg_1", "passage_index": 1, "gold_label": 0},
                {"audit_id": "aud_04", "item_id": "item_neg_2", "passage_index": 2, "gold_label": 0},
            ]

            with open(labels_path, "w", encoding="utf-8") as f:
                for r in fake_labels:
                    f.write(json.dumps(r) + "\n")

            with open(key_path, "w", encoding="utf-8") as f:
                for r in fake_key:
                    f.write(json.dumps(r) + "\n")

            cmd = [
                sys.executable,
                script_path,
                "--labels",
                labels_path,
                "--key",
                key_path,
                "--output",
                output_path,
                "--expected_count",
                "4",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"Process failed: {proc.stderr}")
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check counts
            # Total passages = 4
            self.assertIn("Total passages audited : 4 (2 positive gold label, 2 negative gold label)", content)
            # Total facts = 5
            self.assertIn("Total facts          : 5", content)
            # S = y count: 4 / 5 (80.0%)
            self.assertIn("4 /  5 ( 80.0%)", content)
            # Passages with all S=y & V=y: 2 / 4 (50.0%)
            self.assertIn("2 /  4 ( 50.0%)", content)

            # Check C = y and C = n lines for all three groups
            # Pooled (all 4 passages): C = y 3/4 and C = n 1/4
            # Positive (2 passages): C = y 2/2 and C = n 0/2
            # Negative (2 passages): C = y 1/2 and C = n 1/2
            self.assertIn("4. PASSAGE-LEVEL AUDIT RESULTS", content)
            passage_sec = content.split("4. PASSAGE-LEVEL AUDIT RESULTS")[1]
            pooled_passage, rest = passage_sec.split("Positive source items (2 passages):")
            pos_passage, neg_passage = rest.split("Negative source items (2 passages):")

            self.assertIn("C = y (complete)    :  3 /  4 ( 75.0%)", pooled_passage)
            self.assertIn("C = n (incomplete)  :  1 /  4 ( 25.0%)", pooled_passage)

            self.assertIn("C = y (complete)    :  2 /  2 (100.0%)", pos_passage)
            self.assertIn("C = n (incomplete)  :  0 /  2 (  0.0%)", pos_passage)

            self.assertIn("C = y (complete)    :  1 /  2 ( 50.0%)", neg_passage)
            self.assertIn("C = n (incomplete)  :  1 /  2 ( 50.0%)", neg_passage)

            # Check Wilson interval value for S = y on pooled facts (k=4, n=5)
            low, high = wilson_interval(4, 5)
            expected_ci = f"[{low:.4f}, {high:.4f}]"
            self.assertIn(expected_ci, content)

            # Ensure item IDs and passages are not in output
            self.assertNotIn("item_pos_1", content)
            self.assertNotIn("item_pos_2", content)
            self.assertNotIn("item_neg_1", content)
            self.assertNotIn("item_neg_2", content)


if __name__ == "__main__":
    unittest.main()

