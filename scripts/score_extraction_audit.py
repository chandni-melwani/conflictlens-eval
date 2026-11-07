"""
scripts/score_extraction_audit.py

Read-only scoring of data/extraction_audit_labels.jsonl, joined to key.
Evaluates extraction accuracy and completeness at fact and passage level.
Spec 17.
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.metrics import wilson_interval


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        sys.exit(f"Error: File not found at '{path}'.")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"Error: JSON decode error in '{path}' at line {line_no}: {exc}")
    return records


def validate_and_match(labels_records: list, key_records: list, expected_count: int = 50) -> tuple:
    n_labels = len(labels_records)
    n_key = len(key_records)

    if n_labels != n_key:
        sys.exit(
            f"Error: Line counts differ: labels has {n_labels} records, "
            f"key has {n_key} records."
        )

    if expected_count is not None and expected_count > 0:
        if n_labels != expected_count:
            sys.exit(
                f"Error: Expected {expected_count} records in labels, but got {n_labels}."
            )
        if n_key != expected_count:
            sys.exit(
                f"Error: Expected {expected_count} records in key, but got {n_key}."
            )

    labels_by_id = {}
    for idx, r in enumerate(labels_records, start=1):
        aid = r.get("audit_id")
        if not aid:
            sys.exit(f"Error: Record #{idx} in labels is missing 'audit_id'.")
        if aid in labels_by_id:
            sys.exit(f"Error: Duplicate audit_id '{aid}' in labels.")
        labels_by_id[aid] = r

    key_by_id = {}
    for idx, r in enumerate(key_records, start=1):
        aid = r.get("audit_id")
        if not aid:
            sys.exit(f"Error: Record #{idx} in key is missing 'audit_id'.")
        if aid in key_by_id:
            sys.exit(f"Error: Duplicate audit_id '{aid}' in key.")
        key_by_id[aid] = r

    labels_ids = set(labels_by_id.keys())
    key_ids = set(key_by_id.keys())
    if labels_ids != key_ids:
        missing_in_key = labels_ids - key_ids
        missing_in_labels = key_ids - labels_ids
        errs = []
        if missing_in_key:
            errs.append(f"Audit IDs in labels but missing in key: {sorted(list(missing_in_key))}")
        if missing_in_labels:
            errs.append(f"Audit IDs in key but missing in labels: {sorted(list(missing_in_labels))}")
        sys.exit(f"Error: Audit ID mismatch between labels and key: {'; '.join(errs)}")

    # Validate allowed letters
    allowed_c = {"y", "n", "u"}
    allowed_s = {"y", "n"}
    allowed_va = {"y", "n"}

    for aid, rec in labels_by_id.items():
        c_val = rec.get("C")
        if c_val not in allowed_c:
            sys.exit(
                f"Error: Passage '{aid}' has invalid C answer '{c_val}'. "
                f"Allowed letters: 'y', 'n', 'u'."
            )

        facts = rec.get("facts")
        if not isinstance(facts, list):
            sys.exit(f"Error: Passage '{aid}' facts field must be a list, got {type(facts)}.")

        for f_idx, f in enumerate(facts, start=1):
            if not isinstance(f, dict):
                sys.exit(f"Error: Passage '{aid}' fact #{f_idx} is not a dictionary.")
            s_val = f.get("S")
            if s_val not in allowed_s:
                sys.exit(
                    f"Error: Passage '{aid}' fact #{f_idx} has invalid S answer '{s_val}'. "
                    f"Allowed letters: 'y', 'n'."
                )

            v_val = f.get("V")
            a_val = f.get("A")
            if s_val == "y":
                if v_val not in allowed_va:
                    sys.exit(
                        f"Error: Passage '{aid}' fact #{f_idx} has invalid V answer '{v_val}' for S='y'. "
                        f"Allowed letters: 'y', 'n'."
                    )
                if a_val not in allowed_va:
                    sys.exit(
                        f"Error: Passage '{aid}' fact #{f_idx} has invalid A answer '{a_val}' for S='y'. "
                        f"Allowed letters: 'y', 'n'."
                    )
            else:  # s_val == "n"
                if v_val is not None:
                    sys.exit(
                        f"Error: Passage '{aid}' fact #{f_idx} has non-null V answer '{v_val}' when S='n'. "
                        f"Expected null/None."
                    )
                if a_val is not None:
                    sys.exit(
                        f"Error: Passage '{aid}' fact #{f_idx} has non-null A answer '{a_val}' when S='n'. "
                        f"Expected null/None."
                    )

    return labels_by_id, key_by_id


def compute_group_metrics(labels_list: list) -> dict:
    num_passages = len(labels_list)
    facts_per_passage = [len(l.get("facts", [])) for l in labels_list]
    total_facts = sum(facts_per_passage)

    med_facts = float(statistics.median(facts_per_passage)) if facts_per_passage else 0.0
    max_facts = max(facts_per_passage) if facts_per_passage else 0
    min_facts = min(facts_per_passage) if facts_per_passage else 0
    zero_fact_passages = sum(1 for c in facts_per_passage if c == 0)

    # Fact-level metrics
    all_facts = [f for l in labels_list for f in l.get("facts", [])]
    s_y_count = sum(1 for f in all_facts if f.get("S") == "y")
    s_y_share = (s_y_count / total_facts) if total_facts > 0 else 0.0
    s_y_ci = wilson_interval(s_y_count, total_facts) if total_facts > 0 else (None, None)

    # Among S = y
    v_y_count = sum(1 for f in all_facts if f.get("S") == "y" and f.get("V") == "y")
    v_y_share = (v_y_count / s_y_count) if s_y_count > 0 else 0.0
    v_y_ci = wilson_interval(v_y_count, s_y_count) if s_y_count > 0 else (None, None)

    a_y_count = sum(1 for f in all_facts if f.get("S") == "y" and f.get("A") == "y")
    a_y_share = (a_y_count / s_y_count) if s_y_count > 0 else 0.0
    a_y_ci = wilson_interval(a_y_count, s_y_count) if s_y_count > 0 else (None, None)

    # Passage-level metrics
    all_sv_passages = sum(
        1 for l in labels_list
        if len(l.get("facts", [])) > 0 and all(f.get("S") == "y" and f.get("V") == "y" for f in l.get("facts", []))
    )
    all_sv_share = (all_sv_passages / num_passages) if num_passages > 0 else 0.0
    all_sv_ci = wilson_interval(all_sv_passages, num_passages) if num_passages > 0 else (None, None)

    c_y_count = sum(1 for l in labels_list if l.get("C") == "y")
    c_y_share = (c_y_count / num_passages) if num_passages > 0 else 0.0
    c_y_ci = wilson_interval(c_y_count, num_passages) if num_passages > 0 else (None, None)

    c_n_count = sum(1 for l in labels_list if l.get("C") == "n")
    c_n_share = (c_n_count / num_passages) if num_passages > 0 else 0.0
    c_n_ci = wilson_interval(c_n_count, num_passages) if num_passages > 0 else (None, None)

    c_u_count = sum(1 for l in labels_list if l.get("C") == "u")
    c_u_share = (c_u_count / num_passages) if num_passages > 0 else 0.0
    c_u_ci = wilson_interval(c_u_count, num_passages) if num_passages > 0 else (None, None)

    return {
        "num_passages": num_passages,
        "facts_per_passage": facts_per_passage,
        "total_facts": total_facts,
        "median_facts": med_facts,
        "max_facts": max_facts,
        "min_facts": min_facts,
        "zero_fact_passages": zero_fact_passages,
        "s_y_count": s_y_count,
        "s_y_share": s_y_share,
        "s_y_ci": s_y_ci,
        "v_y_count": v_y_count,
        "v_y_share": v_y_share,
        "v_y_ci": v_y_ci,
        "a_y_count": a_y_count,
        "a_y_share": a_y_share,
        "a_y_ci": a_y_ci,
        "all_sv_passages": all_sv_passages,
        "all_sv_share": all_sv_share,
        "all_sv_ci": all_sv_ci,
        "c_y_count": c_y_count,
        "c_y_share": c_y_share,
        "c_y_ci": c_y_ci,
        "c_n_count": c_n_count,
        "c_n_share": c_n_share,
        "c_n_ci": c_n_ci,
        "c_u_count": c_u_count,
        "c_u_share": c_u_share,
        "c_u_ci": c_u_ci,
    }


def fmt_ci(ci: tuple) -> str:
    if ci[0] is None or ci[1] is None:
        return "undefined"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def format_report(pooled_metrics: dict, pos_metrics: dict, neg_metrics: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("EXTRACTION AUDIT SCORING REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. OVERVIEW & METHODOLOGY")
    lines.append("-" * 72)
    lines.append(
        f"Total passages audited : {pooled_metrics['num_passages']} "
        f"({pos_metrics['num_passages']} positive gold label, {neg_metrics['num_passages']} negative gold label)"
    )
    lines.append("")
    lines.append("Methodological notes:")
    lines.append("- single rater")
    lines.append("- rubric fixed before auditing (reports/extraction_audit_rubric.txt)")
    lines.append("- facts are clustered within passages so fact-level intervals treat them as independent and are optimistic")
    lines.append("- the sample has no 0-fact passage so it cannot estimate total extraction failure")
    lines.append("- and A is a judgement call.")
    lines.append("")
    lines.append(
        "Summary statement: single rater, rubric fixed before auditing "
        "(reports/extraction_audit_rubric.txt), facts are clustered within "
        "passages so fact-level intervals treat them as independent and are "
        "optimistic, the sample has no 0-fact passage so it cannot estimate "
        "total extraction failure, and A is a judgement call."
    )
    lines.append("")
    lines.append("2. FACTS PER PASSAGE DISTRIBUTION")
    lines.append("-" * 72)
    lines.append(f"Pooled (all {pooled_metrics['num_passages']} passages):")
    lines.append(f"  Total facts          : {pooled_metrics['total_facts']}")
    lines.append(f"  Facts per passage    : median = {pooled_metrics['median_facts']:.1f}, max = {pooled_metrics['max_facts']}, min = {pooled_metrics['min_facts']}")
    lines.append(f"  Passages with 0 facts: {pooled_metrics['zero_fact_passages']}")
    lines.append("")
    lines.append(f"Positive source items ({pos_metrics['num_passages']} passages):")
    lines.append(f"  Total facts          : {pos_metrics['total_facts']}")
    lines.append(f"  Facts per passage    : median = {pos_metrics['median_facts']:.1f}, max = {pos_metrics['max_facts']}, min = {pos_metrics['min_facts']}")
    lines.append(f"  Passages with 0 facts: {pos_metrics['zero_fact_passages']}")
    lines.append("")
    lines.append(f"Negative source items ({neg_metrics['num_passages']} passages):")
    lines.append(f"  Total facts          : {neg_metrics['total_facts']}")
    lines.append(f"  Facts per passage    : median = {neg_metrics['median_facts']:.1f}, max = {neg_metrics['max_facts']}, min = {neg_metrics['min_facts']}")
    lines.append(f"  Passages with 0 facts: {neg_metrics['zero_fact_passages']}")
    lines.append("")
    lines.append("3. FACT-LEVEL AUDIT RESULTS")
    lines.append("-" * 72)
    for title, m in [
        (f"Pooled (all {pooled_metrics['num_passages']} passages, n = {pooled_metrics['total_facts']} facts)", pooled_metrics),
        (f"Positive source items ({pos_metrics['num_passages']} passages, n = {pos_metrics['total_facts']} facts)", pos_metrics),
        (f"Negative source items ({neg_metrics['num_passages']} passages, n = {neg_metrics['total_facts']} facts)", neg_metrics),
    ]:
        lines.append(f"{title}:")
        s_cnt, s_tot, s_pct = m["s_y_count"], m["total_facts"], m["s_y_share"] * 100.0
        lines.append(f"  S = y (supported)             : {s_cnt:>2} / {s_tot:>2} ({s_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['s_y_ci'])}")
        v_cnt, v_tot, v_pct = m["v_y_count"], m["s_y_count"], m["v_y_share"] * 100.0
        lines.append(f"  V = y (value faithful, S=y)   : {v_cnt:>2} / {v_tot:>2} ({v_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['v_y_ci'])}")
        a_cnt, a_tot, a_pct = m["a_y_count"], m["s_y_count"], m["a_y_share"] * 100.0
        lines.append(f"  A = y (attr sensible, S=y)    : {a_cnt:>2} / {a_tot:>2} ({a_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['a_y_ci'])}")
        lines.append("")

    lines.append("4. PASSAGE-LEVEL AUDIT RESULTS")
    lines.append("-" * 72)
    for title, m in [
        (f"Pooled (all {pooled_metrics['num_passages']} passages)", pooled_metrics),
        (f"Positive source items ({pos_metrics['num_passages']} passages)", pos_metrics),
        (f"Negative source items ({neg_metrics['num_passages']} passages)", neg_metrics),
    ]:
        lines.append(f"{title}:")
        p_tot = m["num_passages"]
        sv_cnt, sv_pct = m["all_sv_passages"], m["all_sv_share"] * 100.0
        lines.append(f"  All facts S=y & V=y : {sv_cnt:>2} / {p_tot:>2} ({sv_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['all_sv_ci'])}")
        cy_cnt, cy_pct = m["c_y_count"], m["c_y_share"] * 100.0
        lines.append(f"  C = y (complete)    : {cy_cnt:>2} / {p_tot:>2} ({cy_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['c_y_ci'])}")
        cn_cnt, cn_pct = m["c_n_count"], m["c_n_share"] * 100.0
        lines.append(f"  C = n (incomplete)  : {cn_cnt:>2} / {p_tot:>2} ({cn_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['c_n_ci'])}")
        cu_cnt, cu_pct = m["c_u_count"], m["c_u_share"] * 100.0
        lines.append(f"  C = u (unsure)      : {cu_cnt:>2} / {p_tot:>2} ({cu_pct:5.1f}%)  Wilson 95% CI: {fmt_ci(m['c_u_ci'])}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def score_audit(labels_path: str, key_path: str, output_path: str, expected_count: int = 50) -> str:
    labels_records = load_jsonl(labels_path)
    key_records = load_jsonl(key_path)

    labels_by_id, key_by_id = validate_and_match(
        labels_records, key_records, expected_count=expected_count
    )

    pooled_labels = list(labels_by_id.values())
    pos_labels = [labels_by_id[aid] for aid, k in key_by_id.items() if k.get("gold_label") == 1]
    neg_labels = [labels_by_id[aid] for aid, k in key_by_id.items() if k.get("gold_label") == 0]

    pooled_metrics = compute_group_metrics(pooled_labels)
    pos_metrics = compute_group_metrics(pos_labels)
    neg_metrics = compute_group_metrics(neg_labels)

    report_text = format_report(pooled_metrics, pos_metrics, neg_metrics)

    # Sanity check: Ensure no item IDs or passage text are included in the report text
    for k in key_records:
        item_id = k.get("item_id")
        if item_id and item_id in report_text:
            sys.exit(f"Internal error: report text inadvertently contains item_id '{item_id}'.")

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Score extraction audit against key.")
    parser.add_argument(
        "--labels",
        default="data/extraction_audit_labels.jsonl",
        help="Path to extraction audit labels JSONL file.",
    )
    parser.add_argument(
        "--key",
        default="data/extraction_audit_key.jsonl",
        help="Path to extraction audit key JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="reports/extraction_audit_results.txt",
        help="Path to output report text file.",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=50,
        help="Expected record count in labels and key (default: 50, pass 0 to disable count check).",
    )
    args = parser.parse_args()

    report = score_audit(
        labels_path=args.labels,
        key_path=args.key,
        output_path=args.output,
        expected_count=args.expected_count if args.expected_count > 0 else None,
    )
    print(f"Report written to '{args.output}'.")
    print(report)


if __name__ == "__main__":
    main()
