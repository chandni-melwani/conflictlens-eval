import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.cl_detect import check_pair, detect_conflicts, get_candidate_pairs, jaccard_sim
from scripts.cl_normalize import norm_text, numbers, parse_date
from scripts.run_conflictlens import get_out_dir, select_items


class TestConflictLens(unittest.TestCase):
    def test_norm_text(self):
        self.assertEqual(norm_text("The Zorin  Bridge!"), "zorin bridge")

    def test_numbers(self):
        self.assertEqual(numbers("630km (391.5 mi)"), {630.0, 391.5})
        self.assertEqual(numbers("1,200"), {1200.0})

    def test_parse_date(self):
        self.assertEqual(parse_date("10 May 1998"), (1998, 5, 10))
        self.assertEqual(parse_date("May 1998"), (1998, 5, None))
        self.assertEqual(parse_date("1998"), (1998, None, None))
        self.assertIsNone(parse_date("spring"))

    def test_numerical_rule(self):
        f1 = {"entity": "bridge", "attribute": "length", "value": "630km (391.5 mi)"}
        f2 = {"entity": "bridge", "attribute": "length", "value": "391.5"}
        res1 = check_pair(f1, f2)
        self.assertFalse(res1["conflict"])
        self.assertEqual(res1["score"], 0.0)

        f3 = {"entity": "bridge", "attribute": "length", "value": "450.0"}
        f4 = {"entity": "bridge", "attribute": "length", "value": "500.0"}
        res2 = check_pair(f3, f4)
        self.assertTrue(res2["conflict"])
        self.assertEqual(res2["score"], 1.0)

    def test_temporal_rule(self):
        f1 = {"entity": "person", "attribute": "born", "value": "10 May 1998"}
        f2 = {"entity": "person", "attribute": "born", "value": "May 1998"}
        self.assertFalse(check_pair(f1, f2)["conflict"])

        f3 = {"entity": "person", "attribute": "born", "value": "10 May 1998"}
        f4 = {"entity": "person", "attribute": "born", "value": "June 1998"}
        self.assertTrue(check_pair(f3, f4)["conflict"])

        f5 = {"entity": "person", "attribute": "born", "value": "1998"}
        f6 = {"entity": "person", "attribute": "born", "value": "1999"}
        self.assertTrue(check_pair(f5, f6)["conflict"])

    def test_blocking_exact_and_none(self):
        p1_facts = [
            {"entity": "bridge", "attribute": "length", "value": "450 m"},
            {"entity": "bridge", "attribute": "year", "value": "1932"},
        ]
        p2_facts = [
            {"entity": "bridge", "attribute": "length", "value": "500 m"},
            {"entity": "bridge", "attribute": "city", "value": "London"},
            {"entity": "river", "attribute": "length", "value": "100 km"},
        ]
        exact_pairs, n_blocks = get_candidate_pairs(p1_facts, p2_facts, blocking="exact")
        self.assertEqual(len(exact_pairs), 1)
        self.assertEqual(exact_pairs[0], (p1_facts[0], p2_facts[0]))

        none_pairs, _ = get_candidate_pairs(p1_facts, p2_facts, blocking="none")
        self.assertEqual(len(none_pairs), 6)

    def test_fuzzy_matching(self):
        sim1 = jaccard_sim("date of birth", "birth date")
        self.assertAlmostEqual(sim1, 2.0 / 3.0)
        self.assertTrue(sim1 >= 0.5)

        sim2 = jaccard_sim("year", "length")
        self.assertEqual(sim2, 0.0)
        self.assertFalse(sim2 >= 0.5)

    def test_no_candidate_pairs(self):
        res = detect_conflicts([], [], blocking="exact")
        self.assertEqual(res["pred"], 0)
        self.assertEqual(res["score"], 0.0)
        self.assertTrue(res["no_pairs"])

    def test_split_test_with_no_threshold_file_stops_with_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_gold = os.path.join(tmpdir, "gold.jsonl")
            with open(fake_gold, "w", encoding="utf-8") as f:
                f.write('{"id": "dummy", "split": "test", "passage1": "A", "passage2": "B"}\n')
            cmd = [
                sys.executable,
                os.path.join("scripts", "run_conflictlens.py"),
                "--split", "test",
                "--variant", "structured",
                "--blocking", "exact",
                "--gold", fake_gold,
            ]
            old_s = os.path.join("results", "conflictlens_structured_dev", "threshold.json")
            old_s_renamed = None
            if os.path.exists(old_s):
                old_s_renamed = old_s + ".bak"
                os.rename(old_s, old_s_renamed)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("Missing threshold file", proc.stderr + proc.stdout)
            finally:
                if old_s_renamed and os.path.exists(old_s_renamed):
                    os.rename(old_s_renamed, old_s)

    def test_selection_logic(self):
        gold_path, items = select_items("identical")
        self.assertEqual(os.path.normpath(gold_path), os.path.normpath(os.path.join("data", "gold_identical.jsonl")))
        self.assertEqual(len(items), 25)
        self.assertTrue(all(it.get("split") == "same_passage_analysis" for it in items))

        dev_gold, dev_items = select_items("dev")
        self.assertEqual(os.path.normpath(dev_gold), os.path.normpath(os.path.join("data", "gold.jsonl")))
        self.assertEqual(len(dev_items), 51)
        self.assertTrue(all(it.get("split") == "dev" for it in dev_items))

        test_gold, test_items = select_items("test")
        self.assertEqual(os.path.normpath(test_gold), os.path.normpath(os.path.join("data", "gold.jsonl")))
        self.assertEqual(len(test_items), 151)
        self.assertTrue(all(it.get("split") == "test" for it in test_items))

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_gold = os.path.join(tmpdir, "custom.jsonl")
            with open(custom_gold, "w", encoding="utf-8") as f:
                f.write('{"id": "c1", "split": "same_passage_analysis", "label": 1}\n')
                f.write('{"id": "c2", "split": "other", "label": 0}\n')
            c_gold, c_items = select_items("identical", gold=custom_gold)
            self.assertEqual(c_gold, custom_gold)
            self.assertEqual(len(c_items), 1)
            self.assertEqual(c_items[0]["id"], "c1")

            empty_gold = os.path.join(tmpdir, "empty.jsonl")
            with open(empty_gold, "w", encoding="utf-8") as f:
                f.write('{"id": "e1", "split": "other"}\n')
            cmd = [
                sys.executable,
                os.path.join("scripts", "run_conflictlens.py"),
                "--split", "dev",
                "--variant", "structured",
                "--blocking", "exact",
                "--gold", empty_gold,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Zero items selected", proc.stderr)

    def test_non_exact_output_dir_name_contains_blocking_mode(self):
        dir_fuzzy = get_out_dir("structured", "fuzzy", "dev")
        self.assertIn("fuzzy", dir_fuzzy)
        self.assertEqual(os.path.normpath(dir_fuzzy), os.path.normpath(os.path.join("results", "conflictlens_structured_fuzzy_dev")))

        dir_none = get_out_dir("structured", "none", "dev")
        self.assertIn("none", dir_none)
        self.assertEqual(os.path.normpath(dir_none), os.path.normpath(os.path.join("results", "conflictlens_structured_none_dev")))

        dir_exact = get_out_dir("structured", "exact", "dev")
        self.assertEqual(os.path.normpath(dir_exact), os.path.normpath(os.path.join("results", "conflictlens_structured_exact_dev")))

    def test_non_exact_run_with_missing_threshold_file_exits_code_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_gold = os.path.join(tmpdir, "gold.jsonl")
            with open(fake_gold, "w", encoding="utf-8") as f:
                f.write('{"id": "d1", "split": "dev", "passage1": "A", "passage2": "B"}\n')
            cmd = [
                sys.executable,
                os.path.abspath(os.path.join("scripts", "run_conflictlens.py")),
                "--split", "dev",
                "--variant", "structured",
                "--blocking", "fuzzy",
                "--gold", fake_gold,
            ]
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Missing threshold file", proc.stderr + proc.stdout)

    def test_exact_dev_run_with_existing_file_not_overwritten(self):
        import hashlib
        import json
        from scripts.extract import get_cache_key

        with tempfile.TemporaryDirectory() as tmpdir:
            passage = "Albert was born in 1900."
            ckey = get_cache_key(passage, prompt_version="extract_v2")
            cpath = os.path.join(tmpdir, "data", "cache", "extractions", f"{ckey}.json")
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump({
                    "facts": [{"entity": "Albert", "attribute": "birth", "value": "1900"}],
                    "raw": "{}",
                    "cached": True,
                    "parse_ok": True,
                }, f)

            fake_gold = os.path.join(tmpdir, "gold.jsonl")
            with open(fake_gold, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "d1", "split": "dev", "passage1": passage, "passage2": passage, "label": 0}) + "\n")

            thresh_dir = os.path.join(tmpdir, "results", "conflictlens_structured_dev")
            os.makedirs(thresh_dir, exist_ok=True)
            thresh_path = os.path.join(thresh_dir, "threshold.json")
            initial_content = b'{"threshold": 0.05, "mcc": 0.42, "curve": []}\n'
            with open(thresh_path, "wb") as f:
                f.write(initial_content)
            initial_hash = hashlib.sha256(initial_content).hexdigest()

            cmd = [
                sys.executable,
                os.path.abspath(os.path.join("scripts", "run_conflictlens.py")),
                "--split", "dev",
                "--variant", "structured",
                "--blocking", "exact",
                "--gold", fake_gold,
            ]
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Command failed: {proc.stderr}")
            self.assertIn("already exists", proc.stdout + proc.stderr)
            with open(thresh_path, "rb") as f:
                after_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(after_hash, initial_hash)

    def test_fuzzy_dev_run_path_threshold_file_bytes_unchanged(self):
        import hashlib
        import json
        from scripts.extract import get_cache_key

        with tempfile.TemporaryDirectory() as tmpdir:
            passage = "Albert was born in 1900."
            ckey = get_cache_key(passage, prompt_version="extract_v2")
            cpath = os.path.join(tmpdir, "data", "cache", "extractions", f"{ckey}.json")
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump({
                    "facts": [{"entity": "Albert", "attribute": "birth", "value": "1900"}],
                    "raw": "{}",
                    "cached": True,
                    "parse_ok": True,
                }, f)

            fake_gold = os.path.join(tmpdir, "gold.jsonl")
            with open(fake_gold, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "d1", "split": "dev", "passage1": passage, "passage2": passage, "label": 0}) + "\n")

            thresh_dir = os.path.join(tmpdir, "results", "conflictlens_structured_dev")
            os.makedirs(thresh_dir, exist_ok=True)
            thresh_path = os.path.join(thresh_dir, "threshold.json")
            initial_content = b'{"threshold": 0.05, "mcc": 0.42, "curve": []}\n'
            with open(thresh_path, "wb") as f:
                f.write(initial_content)
            initial_hash = hashlib.sha256(initial_content).hexdigest()

            cmd = [
                sys.executable,
                os.path.abspath(os.path.join("scripts", "run_conflictlens.py")),
                "--split", "dev",
                "--variant", "structured",
                "--blocking", "fuzzy",
                "--gold", fake_gold,
            ]
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"Command failed: {proc.stderr}")
            with open(thresh_path, "rb") as f:
                after_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(after_hash, initial_hash)
            out_preds = os.path.join(tmpdir, "results", "conflictlens_structured_fuzzy_dev", "preds.jsonl")
            self.assertTrue(os.path.exists(out_preds))

    def test_non_exact_run_refuses_to_start_if_preds_jsonl_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_gold = os.path.join(tmpdir, "gold.jsonl")
            with open(fake_gold, "w", encoding="utf-8") as f:
                f.write('{"id": "d1", "split": "dev", "passage1": "A", "passage2": "B"}\n')

            out_fuzzy = os.path.join(tmpdir, "results", "conflictlens_structured_fuzzy_dev")
            os.makedirs(out_fuzzy, exist_ok=True)
            with open(os.path.join(out_fuzzy, "preds.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"id": "dummy"}\n')

            thresh_dir = os.path.join(tmpdir, "results", "conflictlens_structured_dev")
            os.makedirs(thresh_dir, exist_ok=True)
            with open(os.path.join(thresh_dir, "threshold.json"), "w", encoding="utf-8") as f:
                f.write('{"threshold": 0.05}\n')

            cmd = [
                sys.executable,
                os.path.abspath(os.path.join("scripts", "run_conflictlens.py")),
                "--split", "dev",
                "--variant", "structured",
                "--blocking", "fuzzy",
                "--gold", fake_gold,
            ]
            proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("already exists", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()

