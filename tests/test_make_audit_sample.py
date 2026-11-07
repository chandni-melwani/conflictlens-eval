import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.extract import get_cache_path
from scripts.make_audit_sample import build_audit_sample


class TestMakeAuditSample(unittest.TestCase):
    def test_build_sample_and_missing_cache_stops_with_error(self):
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "make_audit_sample.py")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gold_path = os.path.join(tmpdir, "gold.jsonl")
            cache_dir = os.path.join(tmpdir, "cache")
            sample_out = os.path.join(tmpdir, "extraction_audit_sample.jsonl")
            key_out = os.path.join(tmpdir, "extraction_audit_key.jsonl")
            plan_out = os.path.join(tmpdir, "extraction_audit_sample_plan.txt")

            os.makedirs(cache_dir, exist_ok=True)

            gold_items = []
            for i in range(1, 31):
                p1_text = f"Positive item {i} passage 1 text"
                p2_text = f"Positive item {i} passage 2 text"
                gold_items.append({
                    "id": f"pos_{i}",
                    "split": "test",
                    "label": 1,
                    "passage1": p1_text,
                    "passage2": p2_text,
                })
                for p_text in (p1_text, p2_text):
                    cp = get_cache_path(p_text, cache_dir=cache_dir, model="qwen2.5:3b", prompt_version="extract_v2")
                    with open(cp, "w", encoding="utf-8") as f:
                        json.dump({"facts": [{"entity": "E", "attribute": "A", "value": "V"}]}, f)

            for i in range(1, 31):
                p1_text = f"Negative item {i} passage 1 text"
                p2_text = f"Negative item {i} passage 2 text"
                gold_items.append({
                    "id": f"neg_{i}",
                    "split": "test",
                    "label": 0,
                    "passage1": p1_text,
                    "passage2": p2_text,
                })
                for p_text in (p1_text, p2_text):
                    cp = get_cache_path(p_text, cache_dir=cache_dir, model="qwen2.5:3b", prompt_version="extract_v2")
                    with open(cp, "w", encoding="utf-8") as f:
                        json.dump({"facts": []}, f)

            with open(gold_path, "w", encoding="utf-8") as f:
                for item in gold_items:
                    f.write(json.dumps(item) + "\n")

            # 1. Successful build
            res = build_audit_sample(
                gold_path=gold_path,
                cache_dir=cache_dir,
                sample_out=sample_out,
                key_out=key_out,
                plan_out=plan_out,
                seed=42,
            )

            # Check: 50 rows in sample file
            with open(sample_out, "r", encoding="utf-8") as f:
                sample_lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(sample_lines), 50)

            # Check: no label or id fields in the sample file
            for row in sample_lines:
                self.assertEqual(set(row.keys()), {"audit_id", "passage_text", "facts"})
                self.assertNotIn("label", row)
                self.assertNotIn("gold_label", row)
                self.assertNotIn("id", row)
                self.assertNotIn("item_id", row)
                self.assertNotIn("split", row)
                self.assertNotIn("pred", row)
                self.assertNotIn("prediction", row)

            # Check key file has 50 rows
            with open(key_out, "r", encoding="utf-8") as f:
                key_lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(key_lines), 50)
            self.assertTrue(os.path.exists(plan_out))

            # 2. Check stop with error when one cache entry is missing
            # Identify the cache path of the first sampled item
            target_item_id = key_lines[0]["item_id"]
            target_p_idx = key_lines[0]["passage_index"]
            target_item = next(it for it in gold_items if it["id"] == target_item_id)
            target_passage = target_item[f"passage{target_p_idx}"]
            target_cache_path = get_cache_path(
                target_passage, cache_dir=cache_dir, model="qwen2.5:3b", prompt_version="extract_v2"
            )
            self.assertTrue(os.path.exists(target_cache_path))
            os.remove(target_cache_path)

            with self.assertRaises(RuntimeError):
                build_audit_sample(
                    gold_path=gold_path,
                    cache_dir=cache_dir,
                    sample_out=sample_out,
                    key_out=key_out,
                    plan_out=plan_out,
                    seed=42,
                )

            cmd = [
                sys.executable,
                script_path,
                "--gold", gold_path,
                "--cache_dir", cache_dir,
                "--sample_out", sample_out,
                "--key_out", key_out,
                "--plan_out", plan_out,
                "--seed", "42",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Missing extraction cache entry", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()

