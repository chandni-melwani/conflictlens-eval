"""
tests/test_audit_extractions.py

Tests for extraction audit tool (Spec 16).
"""

from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestAuditExtractions(unittest.TestCase):
    def test_audit_extractions_flow_format_resume_and_refusal(self):
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "audit_extractions.py")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = os.path.join(tmpdir, "sample.jsonl")
            plan_path = os.path.join(tmpdir, "plan.txt")
            labels_path = os.path.join(tmpdir, "labels.jsonl")

            fake_samples = [
                {
                    "audit_id": "a01",
                    "passage_text": "Alpha Corp acquired Beta Ltd in 2020.",
                    "facts": [
                        {"entity": "Alpha Corp", "attribute": "acquired", "value": "Beta Ltd"},
                        {"entity": "Beta Ltd", "attribute": "acquisition year", "value": "2020"},
                    ],
                },
                {
                    "audit_id": "a02",
                    "passage_text": "Gamma Inc was founded in 1999.",
                    "facts": [
                        {"entity": "Gamma Inc", "attribute": "founded", "value": "1999"},
                    ],
                },
                {
                    "audit_id": "a03",
                    "passage_text": "Delta LLC operates in three regions.",
                    "facts": [],
                },
            ]

            with open(sample_path, "w", encoding="utf-8") as f:
                for row in fake_samples:
                    f.write(json.dumps(row) + "\n")

            sample_hash = compute_sha256(sample_path)
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(f"sample.jsonl: {sample_hash}\n")

            # 1. First run: test rejection of invalid letters, S=n with null V/A, record format, and quit 'q'
            # Inputs:
            # Fact 1:
            #   - 'bad_letter' (invalid) -> re-prompt
            #   - 'y' (S)
            #   - 'invalid' (invalid) -> re-prompt
            #   - 'y' (V)
            #   - 'n' (A)
            # Fact 2:
            #   - 'n' (S) -> V and A not asked
            # Passage C:
            #   - 'invalid' (invalid) -> re-prompt
            #   - 'y' (C)
            # Next passage:
            #   - 'q' (quit)
            stdin_input_1 = "\n".join([
                "bad_letter",
                "y",
                "invalid",
                "y",
                "n",
                "n",
                "invalid",
                "y",
                "q",
            ]) + "\n"

            cmd_1 = [
                sys.executable,
                script_path,
                "--sample", sample_path,
                "--labels", labels_path,
                "--plan", plan_path,
            ]
            proc_1 = subprocess.run(
                cmd_1,
                input=stdin_input_1,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_1.returncode, 0, msg=proc_1.stderr)

            # Check rejection of invalid letters in output
            self.assertIn("Invalid input 'bad_letter'", proc_1.stdout)
            self.assertIn("Invalid input 'invalid'", proc_1.stdout)

            # Check that exactly one record (a01) was written
            self.assertTrue(os.path.exists(labels_path))
            with open(labels_path, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(lines), 1)

            rec_1 = lines[0]
            # Check record format
            self.assertEqual(rec_1["audit_id"], "a01")
            self.assertEqual(len(rec_1["facts"]), 2)
            self.assertEqual(rec_1["facts"][0], {"S": "y", "V": "y", "A": "n"})
            # S is n, so V and A stored as null
            self.assertEqual(rec_1["facts"][1], {"S": "n", "V": None, "A": None})
            self.assertEqual(rec_1["C"], "y")
            # Verify timestamp format
            dt = datetime.fromisoformat(rec_1["timestamp_utc"])
            self.assertIsNotNone(dt)
            self.assertGreaterEqual(rec_1["seconds_spent"], 0.0)

            # 2. Resume behaviour: second run resumes and skips a01
            # For a02:
            #   Fact 1: S='y', V='n', A='y'
            #   C='n'
            # For a03 (0 facts):
            #   C='u'
            stdin_input_2 = "\n".join([
                "y",
                "n",
                "y",
                "n",
                "u",
            ]) + "\n"

            proc_2 = subprocess.run(
                cmd_1,
                input=stdin_input_2,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_2.returncode, 0, msg=proc_2.stderr)

            # Ensure a01 was skipped on resume (item 1 not re-prompted)
            self.assertIn("item 2 of 3", proc_2.stdout)
            self.assertIn("item 3 of 3", proc_2.stdout)

            with open(labels_path, "r", encoding="utf-8") as f:
                resumed_lines = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(resumed_lines), 3)

            self.assertEqual(resumed_lines[0]["audit_id"], "a01")
            self.assertEqual(resumed_lines[1]["audit_id"], "a02")
            self.assertEqual(resumed_lines[1]["facts"], [{"S": "y", "V": "n", "A": "y"}])
            self.assertEqual(resumed_lines[1]["C"], "n")

            self.assertEqual(resumed_lines[2]["audit_id"], "a03")
            self.assertEqual(resumed_lines[2]["facts"], [])
            self.assertEqual(resumed_lines[2]["C"], "u")

            # 3. Refusal of protected paths
            for protected in [
                "extraction_audit_key.jsonl",
                os.path.join("data", "extraction_audit_key.jsonl"),
                "gold.jsonl",
                os.path.join("data", "gold.jsonl"),
                "negative_labels.jsonl",
                os.path.join("data", "negative_labels.jsonl"),
                "preds.jsonl",
                os.path.join("results", "some_dir", "preds.jsonl"),
            ]:
                # Test refusal as input sample path
                proc_refuse_in = subprocess.run(
                    [sys.executable, script_path, "--sample", protected],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc_refuse_in.returncode, 0)
                self.assertIn("Refusing protected", proc_refuse_in.stdout + proc_refuse_in.stderr)

                # Test refusal as output labels path
                proc_refuse_out = subprocess.run(
                    [sys.executable, script_path, "--sample", sample_path, "--labels", protected, "--plan", plan_path],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc_refuse_out.returncode, 0)
                self.assertIn("Refusing protected", proc_refuse_out.stdout + proc_refuse_out.stderr)

            # 4. SHA-256 verification failure stops tool
            bad_plan = os.path.join(tmpdir, "bad_plan.txt")
            with open(bad_plan, "w", encoding="utf-8") as f:
                f.write(f"sample.jsonl: {'0' * 64}\n")

            proc_sha_fail = subprocess.run(
                [sys.executable, script_path, "--sample", sample_path, "--plan", bad_plan],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc_sha_fail.returncode, 0)
            self.assertIn("SHA-256 mismatch", proc_sha_fail.stdout + proc_sha_fail.stderr)


if __name__ == "__main__":
    unittest.main()

