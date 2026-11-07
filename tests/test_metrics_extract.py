import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.metrics_extract import (
    compute_extraction_metrics,
    median_facts,
    median_latency,
    numeric_entity_rate,
    parse_fail_rate,
    verbatim_value_rate,
    zero_fact_rate,
)


class TestMetricsExtract(unittest.TestCase):
    def test_three_passages_hand_computable(self):
        # Three passages: one with 3 facts, one with 0 facts and parse_ok true, one with 0 facts and parse_ok false.
        # Expect zero_fact_rate 1/3 and parse_fail_rate 1/3.
        passages = [
            {
                "parse_ok": True,
                "facts": [
                    {"entity": "E1", "attribute": "A1", "value": "V1"},
                    {"entity": "E2", "attribute": "A2", "value": "V2"},
                    {"entity": "E3", "attribute": "A3", "value": "V3"},
                ],
            },
            {
                "parse_ok": True,
                "facts": [],
            },
            {
                "parse_ok": False,
                "facts": [],
            },
        ]

        self.assertAlmostEqual(parse_fail_rate(passages), 1 / 3, places=6)
        self.assertAlmostEqual(zero_fact_rate(passages), 1 / 3, places=6)
        self.assertEqual(median_facts(passages), 0.0)

    def test_verbatim_value_rate_hand_computable(self):
        # Passage "The bridge is 450 metres long." with values ["450 metres", "500 metres"]. Expect verbatim_value_rate 0.5.
        passage = "The bridge is 450 metres long."
        values = ["450 metres", "500 metres"]

        rate = verbatim_value_rate(passage, values)
        self.assertEqual(rate, 0.5)

        # Also test record format with facts
        records = [
            {
                "passage": passage,
                "facts": [
                    {"entity": "bridge", "attribute": "length", "value": "450 metres"},
                    {"entity": "bridge", "attribute": "length", "value": "500 metres"},
                ],
                "parse_ok": True,
            }
        ]
        self.assertEqual(verbatim_value_rate(records), 0.5)

    def test_numeric_entity_rate_hand_computable(self):
        # Entities ["1528", "Marlow"]. Expect numeric_entity_rate 0.5.
        entities = ["1528", "Marlow"]
        rate = numeric_entity_rate(entities)
        self.assertEqual(rate, 0.5)

        # Also test with record objects
        records = [
            {
                "facts": [
                    {"entity": "1528", "attribute": "year", "value": "1528"},
                    {"entity": "Marlow", "attribute": "location", "value": "Marlow"},
                ]
            }
        ]
        self.assertEqual(numeric_entity_rate(records), 0.5)

    def test_median_latency_computation(self):
        records = [
            {"latency": 2.0, "cached": False},
            {"latency": 4.0, "cached": False},
            {"latency": 10.0, "cached": True},
        ]
        self.assertEqual(median_latency(records), 3.0)

        all_cached = [
            {"latency": 5.0, "cached": True},
            {"latency": 7.0, "cached": True},
        ]
        self.assertEqual(median_latency(all_cached, fallback_to_cached=True), 6.0)

    def test_compute_extraction_metrics_dict(self):
        records = [
            {
                "passage": "The bridge is 450 metres long.",
                "facts": [
                    {"entity": "1528", "attribute": "year", "value": "450 metres"},
                    {"entity": "Marlow", "attribute": "city", "value": "500 metres"},
                ],
                "parse_ok": True,
                "latency": 3.5,
                "cached": False,
            }
        ]
        metrics = compute_extraction_metrics(records)
        self.assertEqual(metrics["parse_fail_rate"], 0.0)
        self.assertEqual(metrics["zero_fact_rate"], 0.0)
        self.assertEqual(metrics["verbatim_value_rate"], 0.5)
        self.assertEqual(metrics["numeric_entity_rate"], 0.5)
        self.assertEqual(metrics["median_facts"], 2.0)
        self.assertEqual(metrics["median_latency"], 3.5)


if __name__ == "__main__":
    unittest.main()
