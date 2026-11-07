import json
import os
import subprocess
import sys
import tempfile
import unittest


class TestScoreRelabel(unittest.TestCase):
    def test_score_relabel_agreement_count_on_fake_inputs(self):
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "score_relabel.py")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_path = os.path.join(tmpdir, "negative_labels.jsonl")
            relabel_path = os.path.join(tmpdir, "relabel_labels.jsonl")
            output_path = os.path.join(tmpdir, "report.txt")

            # Create 20 fake items: 17 agree, 3 disagree
            orig_records = []
            relabel_records = []
            for i in range(1, 21):
                item_id = f"neg_fake_{i}"
                orig_label = "COMPATIBLE"
                # Items 18, 19, 20 disagree
                relabel_label = "CONTRADICT" if i > 17 else "COMPATIBLE"

                orig_records.append({
                    "cand_id": f"fake_{i}",
                    "label": orig_label,
                    "seconds_spent": 10.0,
                })
                relabel_records.append({
                    "id": item_id,
                    "label": relabel_label,
                    "seconds_spent": 12.5 + i,
                })

            with open(orig_path, "w", encoding="utf-8") as f:
                for r in orig_records:
                    f.write(json.dumps(r) + "\n")

            with open(relabel_path, "w", encoding="utf-8") as f:
                for r in relabel_records:
                    f.write(json.dumps(r) + "\n")

            cmd = [
                sys.executable,
                script_path,
                "--relabel",
                relabel_path,
                "--negative_labels",
                orig_path,
                "--output",
                output_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8") as f:
                report_content = f.read()

            self.assertIn("Number agreeing      : 17 / 20", report_content)
            self.assertIn("Percent agreement    : 85.0%", report_content)
            self.assertIn("Count : 3", report_content)
            self.assertIn("neg_fake_18", report_content)
            self.assertIn("neg_fake_19", report_content)
            self.assertIn("neg_fake_20", report_content)


if __name__ == "__main__":
    unittest.main()

