import argparse
import hashlib
import json
import os
import sys

# Import metric functions from the existing metrics module used by evaluate.py
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from metrics import confusion, precision, recall, specificity, f1, balanced_accuracy, mcc

SYSTEMS = [
    ("always_conflict", "always_conflict_dev", "always_conflict_test", None),
    ("always_compatible", "always_compatible_dev", "always_compatible_test", None),
    ("nli_only", "nli_only_dev", "nli_only_test", "nli_only_identical"),
    ("llm_judge", "llm_judge_dev", "llm_judge_test", "llm_judge_identical"),
    ("cl_structured_exact", "conflictlens_structured_exact_dev", "conflictlens_structured_exact_test", "conflictlens_structured_exact_identical"),
    ("cl_hybrid_exact", "conflictlens_hybrid_exact_dev", "conflictlens_hybrid_exact_test", "conflictlens_hybrid_exact_identical"),
    ("cl_structured_fuzzy", "conflictlens_structured_fuzzy_dev", "conflictlens_structured_fuzzy_test", None),
    ("cl_structured_none", "conflictlens_structured_none_dev", "conflictlens_structured_none_test", None),
]


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fmt_metric(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def load_gold_data(gold_path: str, gold_identical_path: str):
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_rows = [json.loads(line) for line in f if line.strip()]

    dev_gold = [r for r in gold_rows if r.get("split") == "dev"]
    test_gold = [r for r in gold_rows if r.get("split") == "test"]

    with open(gold_identical_path, "r", encoding="utf-8") as f:
        id_gold_rows = [json.loads(line) for line in f if line.strip()]
    identical_gold = [r for r in id_gold_rows if r.get("split") == "same_passage_analysis"]

    return dev_gold, test_gold, identical_gold


def load_predictions(preds_path: str, system: str, split: str, gold_items: list[dict], integrity_errors: list[str], synthesized_systems: set[str]):
    gold_ids = [r["id"] for r in gold_items]
    gold_id_set = set(gold_ids)

    if not os.path.exists(preds_path):
        if system == "always_conflict":
            synthesized_systems.add("always_conflict")
            return {gid: 1 for gid in gold_ids}
        elif system == "always_compatible":
            synthesized_systems.add("always_compatible")
            return {gid: 0 for gid in gold_ids}
        else:
            sys.stderr.write(f"Missing input file: {preds_path}\n")
            sys.exit(2)

    with open(preds_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # Integrity checks
    seen_ids = set()
    dup_ids = set()
    invalid_preds = 0
    pred_map = {}

    for item in lines:
        item_id = item.get("id")
        if item_id in seen_ids:
            dup_ids.add(item_id)
        seen_ids.add(item_id)

        pred_val = item.get("pred")
        if pred_val not in (0, 1):
            invalid_preds += 1

        pred_map[item_id] = pred_val

    rel_preds = os.path.relpath(preds_path).replace("\\", "/")
    if dup_ids:
        msg = f"{rel_preds}: {len(dup_ids)} duplicate id(s) found"
        integrity_errors.append(msg)

    missing = gold_id_set - seen_ids
    extra = seen_ids - gold_id_set
    if missing or extra:
        msg = f"{rel_preds}: id set mismatch (missing {len(missing)}, extra {len(extra)})"
        integrity_errors.append(msg)

    if invalid_preds > 0:
        msg = f"{rel_preds}: {invalid_preds} line(s) with pred not in (0, 1)"
        integrity_errors.append(msg)

    return pred_map


def main():
    p = argparse.ArgumentParser(description="Reconcile confusion counts and metrics.")
    p.add_argument("--gold", default=os.path.join("data", "gold.jsonl"), help="Path to gold.jsonl")
    p.add_argument("--gold-identical", default=os.path.join("data", "gold_identical.jsonl"), help="Path to gold_identical.jsonl")
    p.add_argument("--results-dir", default="results", help="Directory containing result folders")
    p.add_argument("--expected", default=os.path.join("reports", "expected_counts.json"), help="Expected counts JSON")
    p.add_argument("--out", default=os.path.join("reports", "count_reconciliation.txt"), help="Output reconciliation report")
    args = p.parse_args()

    # Pre-check input files
    for req_file in [args.gold, args.gold_identical, args.expected]:
        if not os.path.exists(req_file):
            sys.stderr.write(f"Missing input file: {req_file}\n")
            sys.exit(2)

    gold_sha256 = sha256_file(args.gold)
    gold_identical_sha256 = sha256_file(args.gold_identical)

    dev_gold, test_gold, identical_gold = load_gold_data(args.gold, args.gold_identical)

    with open(args.expected, "r", encoding="utf-8") as f:
        expected_entries = json.load(f)

    integrity_errors = []
    synthesized_systems = set()

    # Map of results: key (system, split, subset) -> dict of counts and metrics
    computed_results = {}
    identical_results = []

    for system, dev_folder, test_folder, id_folder in SYSTEMS:
        # Dev split
        dev_preds_path = os.path.join(args.results_dir, dev_folder, "preds.jsonl")
        dev_preds = load_predictions(dev_preds_path, system, "dev", dev_gold, integrity_errors, synthesized_systems)

        # Test split
        test_preds_path = os.path.join(args.results_dir, test_folder, "preds.jsonl")
        test_preds = load_predictions(test_preds_path, system, "test", test_gold, integrity_errors, synthesized_systems)

        # Compute dev subsets
        for subset in ["full", "tok120"]:
            if subset == "full":
                subset_items = dev_gold
            else:
                subset_items = [r for r in dev_gold if r.get("pair_tokens") is not None and r["pair_tokens"] <= 120]

            lbls = [r["label"] for r in subset_items if r["id"] in dev_preds]
            prds = [dev_preds[r["id"]] for r in subset_items if r["id"] in dev_preds]

            n = len(lbls)
            pos = sum(1 for l in lbls if l == 1)
            neg = sum(1 for l in lbls if l == 0)
            tp, fp, fn, tn = confusion(lbls, prds)
            prec = precision(lbls, prds, tp=tp, fp=fp)
            rec = recall(lbls, prds, tp=tp, fn=fn)
            spec = specificity(lbls, prds, tn=tn, fp=fp)
            f1_val = f1(lbls, prds, tp=tp, fp=fp, fn=fn)
            bal_acc = balanced_accuracy(lbls, prds, tp=tp, fp=fp, fn=fn, tn=tn)
            mcc_val = mcc(lbls, prds, tp=tp, fp=fp, fn=fn, tn=tn)

            computed_results[(system, "dev", subset)] = {
                "n": n, "pos": pos, "neg": neg,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "specificity": spec,
                "f1": f1_val, "balanced_accuracy": bal_acc, "mcc": mcc_val
            }

        # Compute test subsets
        for subset in ["full", "tok120"]:
            if subset == "full":
                subset_items = test_gold
            else:
                subset_items = [r for r in test_gold if r.get("pair_tokens") is not None and r["pair_tokens"] <= 120]

            lbls = [r["label"] for r in subset_items if r["id"] in test_preds]
            prds = [test_preds[r["id"]] for r in subset_items if r["id"] in test_preds]

            n = len(lbls)
            pos = sum(1 for l in lbls if l == 1)
            neg = sum(1 for l in lbls if l == 0)
            tp, fp, fn, tn = confusion(lbls, prds)
            prec = precision(lbls, prds, tp=tp, fp=fp)
            rec = recall(lbls, prds, tp=tp, fn=fn)
            spec = specificity(lbls, prds, tn=tn, fp=fp)
            f1_val = f1(lbls, prds, tp=tp, fp=fp, fn=fn)
            bal_acc = balanced_accuracy(lbls, prds, tp=tp, fp=fp, fn=fn, tn=tn)
            mcc_val = mcc(lbls, prds, tp=tp, fp=fp, fn=fn, tn=tn)

            computed_results[(system, "test", subset)] = {
                "n": n, "pos": pos, "neg": neg,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "specificity": spec,
                "f1": f1_val, "balanced_accuracy": bal_acc, "mcc": mcc_val
            }

        # Identical split
        if id_folder is not None:
            id_preds_path = os.path.join(args.results_dir, id_folder, "preds.jsonl")
            id_preds = load_predictions(id_preds_path, system, "identical", identical_gold, integrity_errors, synthesized_systems)
            evaluated_preds = [id_preds[r["id"]] for r in identical_gold if r["id"] in id_preds]
            ones = sum(1 for p in evaluated_preds if p == 1)
            identical_results.append((system, id_folder, len(evaluated_preds), ones))
            computed_results[(system, "identical", "full")] = {
                "n": len(evaluated_preds), "tp": ones, "fp": None, "fn": None, "tn": None, "mcc": None
            }

    # Compare with expected values
    expected_checks = []
    mismatch_count = 0

    for entry in expected_entries:
        sys_key = entry["system"]
        split_key = entry["split"]
        subset_key = entry["subset"]

        computed = computed_results.get((sys_key, split_key, subset_key))
        if computed is None:
            mismatch_count += 1
            expected_checks.append(f"{sys_key:22} {split_key:9} {subset_key:6} : MISMATCH (entry not found in computed results)")
            continue

        mismatches = []
        unchecked = []

        fields = ["n", "tp", "fp", "fn", "tn", "mcc"]
        for field in fields:
            expected_val = entry.get(field)
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

        line_id = f"{sys_key:22} {split_key:9} {subset_key:6} :"
        if mismatches:
            mismatch_count += 1
            details = "; ".join(mismatches)
            if unchecked:
                details += f"; unchecked: {', '.join(unchecked)}"
            expected_checks.append(f"{line_id} MISMATCH ({details})")
        else:
            if unchecked:
                expected_checks.append(f"{line_id} MATCH (unchecked: {', '.join(unchecked)})")
            else:
                expected_checks.append(f"{line_id} MATCH")

    # Add integrity failures to mismatch count
    total_mismatches = mismatch_count + len(integrity_errors)

    # Format output report
    rel_gold = os.path.relpath(args.gold).replace("\\", "/")
    rel_gold_id = os.path.relpath(args.gold_identical).replace("\\", "/")
    rel_results = os.path.relpath(args.results_dir).replace("\\", "/")
    rel_expected = os.path.relpath(args.expected).replace("\\", "/")

    report_lines = []
    report_lines.append("# Count Reconciliation Report")
    report_lines.append("")
    report_lines.append("## Inputs")
    report_lines.append(f"- gold: {rel_gold} (SHA-256: {gold_sha256})")
    report_lines.append(f"- gold-identical: {rel_gold_id} (SHA-256: {gold_identical_sha256})")
    report_lines.append(f"- results-dir: {rel_results}")
    report_lines.append(f"- expected: {rel_expected}")
    report_lines.append("- metrics: all metric functions (confusion, precision, recall, specificity, f1, balanced_accuracy, mcc) imported from scripts.metrics")
    if "always_conflict" in synthesized_systems:
        report_lines.append("- always_conflict: no preds.jsonl in results; constant predictions (1) generated over gold ids of each split")
    if "always_compatible" in synthesized_systems:
        report_lines.append("- always_compatible: no preds.jsonl in results; constant predictions (0) generated over gold ids of each split")
    report_lines.append("")

    report_lines.append("## Integrity Checks")
    if integrity_errors:
        for err in integrity_errors:
            report_lines.append(f"- FAIL: {err}")
    else:
        report_lines.append("All preds.jsonl files read passed integrity checks (unique ids, exact gold id set match, binary pred values). Synthesized baselines are listed under Inputs.")
    report_lines.append("")

    report_lines.append("## Main Table")
    hdr = f"{'System':22} {'Split':5} {'Subset':6} {'n':>4} {'pos':>4} {'neg':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'F1':>7} {'Bal_Acc':>7} {'MCC':>7}"
    report_lines.append(hdr)
    report_lines.append("-" * len(hdr))

    for system, _, _, _ in SYSTEMS:
        for split in ["dev", "test"]:
            for subset in ["full", "tok120"]:
                row = computed_results[(system, split, subset)]
                p_str = fmt_metric(row["precision"])
                r_str = fmt_metric(row["recall"])
                s_str = fmt_metric(row["specificity"])
                f1_str = fmt_metric(row["f1"])
                ba_str = fmt_metric(row["balanced_accuracy"])
                mcc_str = fmt_metric(row["mcc"])
                line = (
                    f"{system:22} {split:5} {subset:6} "
                    f"{row['n']:4d} {row['pos']:4d} {row['neg']:4d} "
                    f"{row['tp']:4d} {row['fp']:4d} {row['fn']:4d} {row['tn']:4d} "
                    f"{p_str:>7} {r_str:>7} {s_str:>7} {f1_str:>7} {ba_str:>7} {mcc_str:>7}"
                )
                report_lines.append(line)

    report_lines.append("")
    report_lines.append("## Identical-Set Table")
    id_hdr = f"{'System':22} {'Folder':40} {'n':>4} {'pred==1':>7}"
    report_lines.append(id_hdr)
    report_lines.append("-" * len(id_hdr))
    for sys_name, folder, n_items, ones in identical_results:
        report_lines.append(f"{sys_name:22} {folder:40} {n_items:4d} {ones:7d}")

    report_lines.append("")
    report_lines.append("## Expected-Value Check")
    for check_line in expected_checks:
        report_lines.append(check_line)

    report_lines.append("")
    if total_mismatches == 0:
        report_lines.append("RESULT: ALL MATCH")
    else:
        report_lines.append(f"RESULT: {total_mismatches} MISMATCHES")

    # Write report
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if total_mismatches == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

