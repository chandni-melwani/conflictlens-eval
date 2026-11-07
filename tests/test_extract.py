import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.extract import (
    EXTRACTION_PROMPT_V2,
    PROMPT_V1,
    PROMPT_V2,
    extract_facts,
    get_cache_key,
    validate_facts,
)


class TestExtract(unittest.TestCase):
    def test_validator_accepts_valid_output(self):
        valid_json = '{"facts": [{"entity": "Eiffel Tower", "attribute": "height", "value": "330m"}]}'
        ok, facts = validate_facts(valid_json)
        self.assertTrue(ok)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["entity"], "Eiffel Tower")

        ok_empty, facts_empty = validate_facts('{"facts": []}')
        self.assertTrue(ok_empty)
        self.assertEqual(facts_empty, [])

    def test_validator_rejects_invalid_json(self):
        ok, _ = validate_facts("not a json string")
        self.assertFalse(ok)
        ok2, _ = validate_facts("{unterminated json")
        self.assertFalse(ok2)

    def test_validator_rejects_missing_facts_key(self):
        ok, _ = validate_facts('{"items": []}')
        self.assertFalse(ok)

    def test_validator_rejects_non_list_facts(self):
        ok, _ = validate_facts('{"facts": "string"}')
        self.assertFalse(ok)
        ok2, _ = validate_facts('{"facts": 123}')
        self.assertFalse(ok2)
        ok3, _ = validate_facts('{"facts": {"entity": "a", "attribute": "b", "value": "c"}}')
        self.assertFalse(ok3)

    def test_validator_rejects_missing_or_empty_field(self):
        ok1, _ = validate_facts('{"facts": [{"entity": "E", "value": "V"}]}')
        self.assertFalse(ok1)
        ok2, _ = validate_facts('{"facts": [{"entity": "", "attribute": "A", "value": "V"}]}')
        self.assertFalse(ok2)
        ok3, _ = validate_facts('{"facts": [{"entity": "E", "attribute": "A", "value": "   "}]}')
        self.assertFalse(ok3)
        ok4, _ = validate_facts('{"facts": [{"entity": 12, "attribute": "A", "value": "V"}]}')
        self.assertFalse(ok4)

    def test_cache_key_stable_and_version_sensitive(self):
        k1 = get_cache_key("A simple passage", "qwen2.5:3b", "extract_v1")
        k2 = get_cache_key("A simple passage", "qwen2.5:3b", "extract_v1")
        self.assertEqual(k1, k2)

        k3 = get_cache_key("A simple passage", "qwen2.5:3b", "extract_v2")
        self.assertNotEqual(k1, k3)

    def test_cache_hit_does_not_call_network(self):
        calls = 0

        def fake_network(prompt):
            nonlocal calls
            calls += 1
            return '{"facts": [{"entity": "Fake", "attribute": "status", "value": "active"}]}'

        with tempfile.TemporaryDirectory() as tmp_dir:
            res1 = extract_facts("Test passage", cache_dir=tmp_dir, network_caller=fake_network)
            self.assertEqual(calls, 1)
            self.assertFalse(res1.get("cached", False))
            self.assertEqual(len(res1["facts"]), 1)

            def fail_network(prompt):
                raise AssertionError("Network caller should not have been called on cache hit!")

            res2 = extract_facts("Test passage", cache_dir=tmp_dir, network_caller=fail_network)
            self.assertEqual(calls, 1)
            self.assertTrue(res2.get("cached", False))
            self.assertEqual(res2["facts"], res1["facts"])

    def test_extract_v2_prompt_and_calling(self):
        # Verify v2 prompt text contains required rules and examples
        self.assertIn("You read a passage and list its facts as (entity, attribute, value) triples.", PROMPT_V2)
        self.assertIn("Passage: The Zorin Bridge opened in 1932 and is 450 metres long.", PROMPT_V2)
        self.assertIn("Passage: After the 1887 flood, the town of Marlow lost 40 of its 120 homes.", PROMPT_V2)
        self.assertIn("Passage: Stage | Date | Winner 1 | 3 May | A. Roe (ENG)", PROMPT_V2)

        received_prompts = []

        def fake_network(prompt):
            received_prompts.append(prompt)
            return '{"facts": [{"entity": "Zorin Bridge", "attribute": "opening year", "value": "1932"}]}'

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = extract_facts(
                "My test passage for v2",
                cache_dir=tmp_dir,
                prompt_version="extract_v2",
                network_caller=fake_network,
            )
            self.assertEqual(len(received_prompts), 1)
            self.assertIn("My test passage for v2", received_prompts[0])
            self.assertEqual(res["prompt_version"], "extract_v2")
            self.assertEqual(len(res["facts"]), 1)


if __name__ == "__main__":
    unittest.main()

