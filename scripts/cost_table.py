import argparse
import hashlib
import json
import os
import sys

# Import metric functions from scripts.metrics
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from metrics import confusion, mcc

SYSTEMS = [
    ("always_conflict", None, "always_conflict_test"),
    ("nli_only", None, "nli_only_test"),
    ("llm_judge", None, "llm_judge_test"),
    ("cl_structured_exact", "conflictlens_structured_exact_test", "conflictlens_structured_exact_test"),
    ("cl_structured_fuzzy", "conflictlens_structured_fuzzy_test", "conflictlens_structured_fuzzy_test"),
    ("cl_structured_none", "conflictlens_structured_none_test", "conflictlens_structured_none_test"),
    ("cl_hybrid_exact", "conflictlens_hybrid_exact_test", "conflictlens_hybrid_exact_test"),
]

ALLOWED_DETAILS_FIELDS = {
    "id",
    "no_pairs",
    "fallback_used",
    "n_candidate_pairs",
    "p1_ok",
    "p2_ok",
}


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_gold_test(gold_path: str):
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_rows = []
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Only read id, split, label
            if raw.get("split") == "test":
                gold_rows.append({
                    "id": raw["id"],
                    "split": raw["split"],
                    "label": raw["label"],
                })
    return gold_rows


def load_predictions(preds_path: str, system: str, gold_test_items: list[dict], integrity_errors: list[str]):
    gold_ids = [r["id"] for r in gold_test_items]
    gold_id_set = set(gold_ids)

    if system == "always_conflict":
        # Constant predictions (1) over gold test ids
        return {gid: 1 for gid in gold_ids}

    if not os.path.exists(preds_path):
        sys.stderr.write(f"Missing input file: {preds_path}\n")
        sys.exit(2)

    seen_ids = set()
    dup_ids = set()
    invalid_preds = 0
    pred_map = {}

    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            item_id = raw.get("id")
            if item_id in seen_ids:
                dup_ids.add(item_id)
            seen_ids.add(item_id)

            pred_val = raw.get("pred")
            if pred_val not in (0, 1):
                invalid_preds += 1

            pred_map[item_id] = pred_val

    rel_preds = os.path.relpath(preds_path).replace("\\", "/")
    if dup_ids:
        integrity_errors.append(f"{rel_preds}: {len(dup_ids)} duplicate id(s) found")

    missing = gold_id_set - seen_ids
    extra = seen_ids - gold_id_set
    if missing or extra:
        integrity_errors.append(f"{rel_preds}: id set mismatch (missing {len(missing)}, extra {len(extra)})")

    if invalid_preds > 0:
        integrity_errors.append(f"{rel_preds}: {invalid_preds} line(s) with pred not in (0, 1)")

    return pred_map


def load_details(details_path: str, gold_test_items: list[dict], integrity_errors: list[str]):
    gold_ids = [r["id"] for r in gold_test_items]
    gold_id_set = set(gold_ids)

    if not os.path.exists(details_path):
        sys.stderr.write(f"Missing input file: {details_path}\n")
        sys.exit(2)

    seen_ids = set()
    dup_ids = set()
    details_rows = []

    with open(details_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Strictly read only allowed fields
            filtered = {k: raw[k] for k in ALLOWED_DETAILS_FIELDS if k in raw}

            item_id = filtered.get("id")
            if item_id in seen_ids:
                dup_ids.add(item_id)
            seen_ids.add(item_id)

            details_rows.append(filtered)

    rel_details = os.path.relpath(details_path).replace("\\", "/")
    if dup_ids:
        integrity_errors.append(f"{rel_details}: {len(dup_ids)} duplicate id(s) found")

    missing = gold_id_set - seen_ids
    extra = seen_ids - gold_id_set
    if missing or extra:
        integrity_errors.append(f"{rel_details}: id set mismatch (missing {len(missing)}, extra {len(extra)})")

    return details_rows


def main():
    p = argparse.ArgumentParser(description="Derive cost columns from details.jsonl and predictions.")
    p.add_argument("--results-dir", default="results", help="Directory containing result folders")
    p.add_argument("--gold", default=os.path.join("data", "gold.jsonl"), help="Path to gold.jsonl")
    p.add_argument("--expected", default=os.path.join("reports", "expected_cost.json"), help="Expected cost JSON")
    p.add_argument("--out", default=os.path.join("reports", "cost_table.txt"), help="Output cost table report")
    args = p.parse_args()

    # Pre-check non-system input files
    for req_file in [args.gold, args.expected]:
        if not os.path.exists(req_file):
            sys.stderr.write(f"Missing input file: {req_file}\n")
            sys.exit(2)

    # Pre-check existence of details files and preds files for required systems
    for sys_name, details_folder, preds_folder in SYSTEMS:
        if details_folder is not None:
            det_path = os.path.join(args.results_dir, details_folder, "details.jsonl")
            if not os.path.exists(det_path):
                sys.stderr.write(f"Missing input file: {det_path}\n")
                sys.exit(2)
        if preds_folder is not None and sys_name != "always_conflict":
            prd_path = os.path.join(args.results_dir, preds_folder, "preds.jsonl")
            if not os.path.exists(prd_path):
                sys.stderr.write(f"Missing input file: {prd_path}\n")
                sys.exit(2)

    gold_sha256 = sha256_file(args.gold)
    gold_test = load_gold_test(args.gold)

    with open(args.expected, "r", encoding="utf-8") as f:
        expected_entries = json.load(f)

    integrity_errors = []
    computed_rows = []
    computed_by_system = {}

    for sys_name, details_folder, preds_folder in SYSTEMS:
        # Load preds and compute MCC
        if preds_folder is not None and sys_name != "always_conflict":
            preds_path = os.path.join(args.results_dir, preds_folder, "preds.jsonl")
        else:
            preds_path = None
        preds_map = load_predictions(preds_path, sys_name, gold_test, integrity_errors)

        lbls = [r["label"] for r in gold_test if r["id"] in preds_map]
        prds = [preds_map[r["id"]] for r in gold_test if r["id"] in preds_map]
        tp, fp, fn, tn = confusion(lbls, prds)
        mcc_val = mcc(lbls, prds, tp=tp, fp=fp, fn=fn, tn=tn)

        n_items = len(gold_test)

        # LLM calls per item
        if sys_name in ("always_conflict", "nli_only"):
            llm_calls = 0
        elif sys_name == "llm_judge":
            llm_calls = 1
        else:
            llm_calls = 2

        # Process details if available
        if details_folder is not None:
            details_path = os.path.join(args.results_dir, details_folder, "details.jsonl")
            details_rows = load_details(details_path, gold_test, integrity_errors)

            pairs_total = sum(r.get("n_candidate_pairs", 0) for r in details_rows)
            items_with_pairs = sum(1 for r in details_rows if r.get("n_candidate_pairs", 0) > 0)
            no_pair_items = n_items - items_with_pairs
            max_pairs = max((r.get("n_candidate_pairs", 0) for r in details_rows), default=0)
            parse_failures = (
                sum(1 for r in details_rows if not r.get("p1_ok", True))
                + sum(1 for r in details_rows if not r.get("p2_ok", True))
            )
            fallback_count = sum(1 for r in details_rows if r.get("fallback_used") is True)
        else:
            details_rows = None
            pairs_total = None
            items_with_pairs = None
            no_pair_items = None
            max_pairs = None
            parse_failures = None
            fallback_count = 0

        # NLI full-passage calls
        if sys_name == "nli_only":
            nli_full_calls = n_items
        elif sys_name == "cl_hybrid_exact":
            nli_full_calls = fallback_count
        else:
            nli_full_calls = 0

        row_dict = {
            "system": sys_name,
            "items": n_items,
            "mcc": mcc_val,
            "llm_calls_per_item": llm_calls,
            "nli_full_calls": nli_full_calls,
            "pairs_total": pairs_total,
            "items_with_pairs": items_with_pairs,
            "no_pair_items": no_pair_items,
            "max_pairs": max_pairs,
            "extraction_parse_failures": parse_failures,
        }
        computed_rows.append(row_dict)
        computed_by_system[sys_name] = row_dict

    # Check against expected values
    expected_checks = []
    mismatch_count = 0

    for entry in expected_entries:
        sys_key = entry.get("system")
        computed = computed_by_system.get(sys_key)
        line_id = f"{sys_key:22} :"

        if computed is None:
            mismatch_count += 1
            expected_checks.append(f"{line_id} MISMATCH (system not found in computed results)")
            continue

        mismatches = []
        unchecked = []

        for field, expected_val in entry.items():
            if field == "system":
                continue
            if expected_val is None:
                unchecked.append(field)
            else:
                computed_val = computed.get(field)
                if computed_val is None:
                    mismatches.append(f"{field}: expected={expected_val}, computed=None")
                elif field == "mcc":
                    if abs(computed_val - expected_val) > 0.0005:
                        mismatches.append(f"mcc: expected={expected_val:.4f}, computed={computed_val:.4f}")
                else:
                    if computed_val != expected_val:
                        mismatches.append(f"{field}: expected={expected_val}, computed={computed_val}")

        if mismatches:
            mismatch_count += 1
            details_str = "; ".join(mismatches)
            if unchecked:
                details_str += f"; unchecked: {', '.join(unchecked)}"
            expected_checks.append(f"{line_id} MISMATCH ({details_str})")
        else:
            if unchecked:
                expected_checks.append(f"{line_id} MATCH (unchecked: {', '.join(unchecked)})")
            else:
                expected_checks.append(f"{line_id} MATCH")

    total_mismatches = mismatch_count + len(integrity_errors)

    # Format report
    rel_gold = os.path.relpath(args.gold).replace("\\", "/")
    rel_results = os.path.relpath(args.results_dir).replace("\\", "/")
    rel_expected = os.path.relpath(args.expected).replace("\\", "/")

    report_lines = []
    report_lines.append("# Cost Table Report")
    report_lines.append("")
    report_lines.append("## Inputs")
    report_lines.append(f"- gold: {rel_gold} (SHA-256: {gold_sha256})")
    report_lines.append(f"- results-dir: {rel_results}")
    report_lines.append(f"- expected: {rel_expected}")
    report_lines.append("- split: test (full subset, 151 items)")
    report_lines.append("- metrics: MCC computed from results/<folder>/preds.jsonl against gold with scripts.metrics")
    report_lines.append("- always_conflict: constant predictions (1) generated over gold test ids")
    report_lines.append("")

    report_lines.append("## Integrity Checks")
    if integrity_errors:
        for err in integrity_errors:
            report_lines.append(f"- FAIL: {err}")
    else:
        report_lines.append("All preds.jsonl and details.jsonl files read passed integrity checks (unique ids, exact gold test id set match, binary pred values). Constant baselines are listed under Inputs.")
    report_lines.append("")

    report_lines.append("## Cost Table")
    headers = [
        ("System", "<22"),
        ("MCC", ">7"),
        ("LLM calls per item", ">18"),
        ("NLI full-passage calls", ">22"),
        ("Pair comparisons (upper bound on NLI calls)", ">43"),
        ("Items with at least one pair", ">28"),
        ("Max pairs in one item", ">21"),
        ("Extraction parse failures", ">25"),
        ("Items", ">7"),
    ]
    hdr = "  ".join(f"{h:{fmt}}" for h, fmt in headers)
    report_lines.append(hdr)
    report_lines.append("-" * len(hdr))

    for row in computed_rows:
        p_comp = "n/a" if row["pairs_total"] is None else f"{row['pairs_total']:d}"
        p_items = "n/a" if row["items_with_pairs"] is None else f"{row['items_with_pairs']:d}"
        max_p = "n/a" if row["max_pairs"] is None else f"{row['max_pairs']:d}"
        ext_fail = "n/a" if row["extraction_parse_failures"] is None else f"{row['extraction_parse_failures']:d}"
        mcc_str = "n/a" if row["mcc"] is None else f"{row['mcc']:.4f}"

        vals = [
            (row["system"], "<22"),
            (mcc_str, ">7"),
            (str(row["llm_calls_per_item"]), ">18"),
            (str(row["nli_full_calls"]), ">22"),
            (p_comp, ">43"),
            (p_items, ">28"),
            (max_p, ">21"),
            (ext_fail, ">25"),
            (str(row["items"]), ">7"),
        ]
        line = "  ".join(f"{v:{fmt}}" for v, fmt in vals)
        report_lines.append(line)

    report_lines.append("")
    report_lines.append("Footnotes:")
    report_lines.append("- by design; extraction calls were served from cache in ablation runs")
    report_lines.append(
        "- Numerical and temporal pairs are decided without NLI, so NLI pair calls are at most this number. "
        "detector_types in details.jsonl lists flagged pairs only, so the exact NLI pair count was not recorded."
    )
    report_lines.append("")

    report_lines.append("## Expected-Value Check")
    for check_line in expected_checks:
        report_lines.append(check_line)

    report_lines.append("")
    if total_mismatches == 0:
        report_lines.append("RESULT: ALL MATCH")
    else:
        report_lines.append(f"RESULT: {total_mismatches} MISMATCHES")

    # Write output report
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if total_mismatches == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
