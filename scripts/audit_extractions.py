"""
scripts/audit_extractions.py

Extraction audit tool and frozen rubric (Spec 16).
Hand-audit tool for the 50 sampled passages.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RUBRIC = """Per extracted fact, answers y or n:
- S, supported: y if the fact's claim is stated in the passage (paraphrase is fine), n if the extractor added or distorted it.
- V, value faithful: y if the value keeps every number, date, name and the meaning of the passage, n if any is altered, dropped or wrongly merged. Asked only when S is y; otherwise stored as null.
- A, attribute sensible: y if another passage giving the same kind of information about the same entity would plausibly get the same attribute name, n if the attribute is a passage-specific verb or phrase. Asked only when S is y; otherwise null.
Per passage, one answer y, n or u:
- C, complete: y if every contradiction-relevant item in the passage (number, date, named state, quantity) appears in some extracted fact, n if at least one is missing, u if unsure.
A passage with 0 extracted facts gets only the C question.
Notes: single rater, rubric fixed before auditing, judgement-based. Do not describe it as an objective measure."""

PROTECTED_BASENAMES = {
    "extraction_audit_key.jsonl",
    "negative_labels.jsonl",
    "gold.jsonl",
}


def is_protected_path(path: str) -> bool:
    if not path:
        return False
    norm = os.path.normpath(path).replace("\\", "/").lower()
    base = os.path.basename(norm)
    if base in PROTECTED_BASENAMES:
        return True
    if "preds" in base:
        return True
    return False


def validate_not_protected(path: str, role: str = "path"):
    if is_protected_path(path):
        sys.exit(f"Error: Refusing protected {role} path: '{path}'")


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_expected_hash_from_plan(plan_path: str, sample_path: str) -> str:
    if not os.path.exists(plan_path):
        sys.exit(f"Error: Plan file not found at '{plan_path}'")
    with open(plan_path, "r", encoding="utf-8") as f:
        content = f.read()

    sample_name = os.path.basename(sample_path)
    # Match lines like "data/extraction_audit_sample.jsonl: <hash>" or "<filename>: <hash>"
    pattern = re.compile(rf"{re.escape(sample_name)}:\s*([a-fA-F0-9]{{64}})")
    match = pattern.search(content)
    if match:
        return match.group(1).lower()

    # Match extraction_audit_sample.jsonl in plan
    match = re.search(r"extraction_audit_sample\.jsonl:\s*([a-fA-F0-9]{64})", content)
    if match:
        return match.group(1).lower()

    # Match any 64-character hex string if single or first in file
    match = re.search(r"\b([a-fA-F0-9]{64})\b", content)
    if match:
        return match.group(1).lower()

    sys.exit(f"Error: Could not find expected SHA-256 for '{sample_name}' in plan file '{plan_path}'")


def verify_sample_hash(sample_path: str, plan_path: str):
    expected_hash = extract_expected_hash_from_plan(plan_path, sample_path)
    actual_hash = compute_sha256(sample_path)
    if actual_hash.lower() != expected_hash.lower():
        sys.exit(
            f"Error: SHA-256 mismatch for sample file '{sample_path}'.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )


def load_sample_file(sample_path: str) -> list[dict]:
    if not os.path.exists(sample_path):
        sys.exit(f"Error: Sample file not found at '{sample_path}'")

    records = []
    with open(sample_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def load_completed_ids(labels_path: str) -> set[str]:
    completed = set()
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "audit_id" in rec:
                        completed.add(rec["audit_id"])
                except json.JSONDecodeError:
                    pass
    return completed


def ask_choice(prompt: str, valid_choices: set[str]) -> str:
    valid_with_q = valid_choices | {"q"}
    while True:
        try:
            val = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if val in valid_with_q:
            return val
        print(f"Invalid input '{val}'. Valid choices: {', '.join(sorted(valid_choices))}, q to quit.")


def run_audit(sample_path: str, labels_path: str, plan_path: str):
    # Refuse protected paths
    validate_not_protected(sample_path, role="sample")
    validate_not_protected(labels_path, role="output labels")
    validate_not_protected(plan_path, role="plan")

    # Verify SHA-256 against plan file
    verify_sample_hash(sample_path, plan_path)

    # Print rubric at tool start
    print(RUBRIC)
    print("\n" + "=" * 60 + "\n")

    sample_records = load_sample_file(sample_path)
    completed_ids = load_completed_ids(labels_path)

    total_items = len(sample_records)
    remaining = [r for r in sample_records if r.get("audit_id") not in completed_ids]
    if not remaining:
        print(f"All {total_items} items already audited in '{labels_path}'.")
        return

    for idx, item in enumerate(sample_records, start=1):
        audit_id = item.get("audit_id")
        if audit_id in completed_ids:
            continue

        print(f"\nitem {idx} of {total_items}")
        print(f"Passage:\n{item.get('passage_text', '')}\n")

        facts = item.get("facts", [])
        if facts:
            print("Facts:")
            for f_idx, fact in enumerate(facts, 1):
                ent = fact.get("entity", "")
                attr = fact.get("attribute", "")
                val = fact.get("value", "")
                print(f"  {f_idx}. Entity: {ent} | Attribute: {attr} | Value: {val}")
        else:
            print("Facts: (0 extracted facts)")

        t0 = time.time()
        fact_answers = []
        quit_requested = False

        for f_idx, fact in enumerate(facts, 1):
            s_prompt = f"  Fact {f_idx} - S (supported? y/n, q to quit): "
            s_ans = ask_choice(s_prompt, {"y", "n"})
            if s_ans == "q":
                quit_requested = True
                break

            if s_ans == "y":
                v_prompt = f"  Fact {f_idx} - V (value faithful? y/n, q to quit): "
                v_ans = ask_choice(v_prompt, {"y", "n"})
                if v_ans == "q":
                    quit_requested = True
                    break

                a_prompt = f"  Fact {f_idx} - A (attribute sensible? y/n, q to quit): "
                a_ans = ask_choice(a_prompt, {"y", "n"})
                if a_ans == "q":
                    quit_requested = True
                    break

                fact_answers.append({"S": "y", "V": v_ans, "A": a_ans})
            else:
                # S is n: V and A stored as null
                fact_answers.append({"S": "n", "V": None, "A": None})

        if quit_requested:
            print("Quit requested. Exiting with no edits.")
            return

        c_prompt = "Passage - C (complete? y/n/u, q to quit): "
        c_ans = ask_choice(c_prompt, {"y", "n", "u"})
        if c_ans == "q":
            print("Quit requested. Exiting with no edits.")
            return

        sec_spent = round(time.time() - t0, 2)
        record = {
            "audit_id": audit_id,
            "facts": fact_answers,
            "C": c_ans,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seconds_spent": sec_spent,
        }

        out_dir = os.path.dirname(labels_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        completed_ids.add(audit_id)

    print(f"\nAudit finished. All {total_items} items completed.")


def main():
    parser = argparse.ArgumentParser(description="Extraction audit tool (Spec 16)")
    parser.add_argument(
        "--sample",
        default=os.path.join("data", "extraction_audit_sample.jsonl"),
        help="Path to sample file",
    )
    parser.add_argument(
        "--labels",
        default=os.path.join("data", "extraction_audit_labels.jsonl"),
        help="Path to output labels file",
    )
    parser.add_argument(
        "--plan",
        default=os.path.join("reports", "extraction_audit_sample_plan.txt"),
        help="Path to sample plan file",
    )
    args = parser.parse_args()

    run_audit(
        sample_path=args.sample,
        labels_path=args.labels,
        plan_path=args.plan,
    )


if __name__ == "__main__":
    main()

