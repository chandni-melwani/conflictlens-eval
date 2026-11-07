import math
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.run_baselines import fill_judge_prompt, find_best_threshold
from scripts.style_checks import starts_with_pronoun


class TestBaselines(unittest.TestCase):
    def test_threshold_selection(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        best_t, best_mcc, curve = find_best_threshold(labels, scores)
        self.assertEqual(best_t, 0.15)
        expected_mcc = 2.0 / math.sqrt(12)
        self.assertAlmostEqual(best_mcc, expected_mcc, places=4)

    def test_running_with_split_test_and_no_threshold_file_stops_with_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_thresh = os.path.join(tmpdir, "missing_thresh.json")
            cmd = [
                sys.executable,
                os.path.join("scripts", "run_baselines.py"),
                "--split",
                "test",
                "--threshold_file",
                fake_thresh,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            combined_output = proc.stderr + proc.stdout
            self.assertIn("Missing threshold file", combined_output)

    def test_pronoun_check(self):
        self.assertTrue(starts_with_pronoun("He was born in 1990."))
        self.assertFalse(starts_with_pronoun("Hector was born in 1990."))
        # Additional checks for other specified pronouns / demonstratives
        self.assertTrue(starts_with_pronoun("They were colleagues."))
        self.assertFalse(starts_with_pronoun("Theresa was born in 1990."))
        self.assertTrue(starts_with_pronoun("This is a statement."))
        self.assertFalse(starts_with_pronoun("Thistle is a plant."))

    def test_judge_prompt_filler_leaves_braces_intact(self):
        p1 = "First passage text."
        p2 = "Second passage text."
        prompt = fill_judge_prompt(p1, p2)
        self.assertIn('{"contradiction": true} or {"contradiction": false}', prompt)
        self.assertIn("Passage 1: First passage text.", prompt)
        self.assertIn("Passage 2: Second passage text.", prompt)


if __name__ == "__main__":
    unittest.main()

