import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.nli import contradiction_score, get_nli_components, nli_probs


class TestNLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.model, cls.id2label = get_nli_components()

    def test_id2label_names(self):
        labels = set(self.id2label.values())
        expected = {"contradiction", "entailment", "neutral"}
        self.assertEqual(labels, expected, f"Expected {expected}, got {labels}")

    def test_sanity_check_entailment(self):
        premise = "A man is eating pizza"
        hypothesis = "A man eats something"
        probs = nli_probs(premise, hypothesis)
        self.assertGreater(
            probs["entailment"],
            probs["contradiction"],
            f"Sanity check failed: P(entailment)={probs['entailment']} not greater than P(contradiction)={probs['contradiction']} for ('{premise}', '{hypothesis}')",
        )

    def test_sanity_check_contradiction(self):
        a = "The shop is open."
        b = "The shop is closed."
        probs = nli_probs(a, b)
        self.assertGreater(
            probs["contradiction"],
            probs["entailment"],
            f"Sanity check failed: P(contradiction)={probs['contradiction']} not greater than P(entailment)={probs['entailment']} for ('{a}', '{b}')",
        )
        score = contradiction_score(a, b)
        self.assertGreater(score, 0.5, f"Expected high contradiction score, got {score}")


if __name__ == "__main__":
    unittest.main()

