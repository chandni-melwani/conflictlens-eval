import os
import subprocess
import sys
import unittest


class TestStyleScores(unittest.TestCase):
    def test_style_scores_output_has_151_lines(self):
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "style_checks.py"))
        proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        preds_path = os.path.join("results", "style_only_test", "preds.jsonl")
        self.assertTrue(os.path.isfile(preds_path), f"Missing output file: {preds_path}")
        with open(preds_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 151)


if __name__ == "__main__":
    unittest.main()

