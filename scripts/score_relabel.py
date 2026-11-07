import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.metrics import wilson_interval


def load_negative_labels(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"Error: Original negative labels file not found at '{path}'.")
    labels_by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" in rec and rec["id"] is not None:
                str_id = str(rec["id"])
                labels_by_id[str_id] = rec
                if str_id.startswith("neg_"):
                    labels_by_id[str_id[4:]] = rec
                else:
                    labels_by_id[f"neg_{str_id}"] = rec
            if "cand_id" in rec and rec["cand_id"] is not None:
                cid = str(rec["cand_id"])
                labels_by_id[cid] = rec
                if cid.startswith("neg_"):
                    labels_by_id[cid[4:]] = rec
                else:
                    labels_by_id[f"neg_{cid}"] = rec
    return labels_by_id


def load_relabel_labels(path: str) -> list:
    if not os.path.exists(path):
        sys.exit(f"Error: Relabel labels file not found at '{path}'.")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_report(n, agree_count, agree_pct, ci_low, ci_high,
                  disagreements, orig_counts, med_sec,
                  relabel_path, negative_path, expected_count):
    lines = []
    lines.append("=" * 70)
    lines.append("INTRA-ANNOTATOR RE-LABEL AGREEMENT REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Note: This is intra-annotator agreement on COMPATIBLE-only items,")
    lines.append("not kappa and not inter-annotator agreement.")
    lines.append("")
    lines.append("1. SAMPLE & INPUTS")
    lines.append("-" * 70)
    lines.append(f"Relabel source   : {relabel_path}")
    lines.append(f"Original source  : {negative_path}")
    lines.append(f"Sample size (n)  : {n}")
    if expected_count is not None:
        lines.append(f"Line count check : {expected_count} expected, {n} observed (OK)")
    lines.append(f"ID matching      : {n}/{n} IDs matched to active original labels (OK)")
    lines.append("")
    lines.append("2. ORIGINAL LABEL DISTRIBUTION (IN SAMPLE)")
    lines.append("-" * 70)
    for lbl, count in sorted(orig_counts.items(), key=lambda x: (-x[1], x[0])):
        pct = (count / n) * 100.0 if n > 0 else 0.0
        lines.append(f"{lbl:<12} : {count} ({pct:.1f}%)")
    lines.append(f"{'Total':<12} : {n}")
    lines.append("")
    lines.append("3. AGREEMENT")
    lines.append("-" * 70)
    lines.append(f"Number agreeing      : {agree_count} / {n}")
    lines.append(f"Percent agreement    : {agree_pct:.1f}%")
    if ci_low is not None and ci_high is not None:
        lines.append(f"Wilson 95% interval  : [{ci_low:.4f}, {ci_high:.4f}] ({ci_low * 100.0:.1f}% - {ci_high * 100.0:.1f}%)")
    else:
        lines.append("Wilson 95% interval  : undefined")
    lines.append("")
    lines.append("4. DISAGREEMENTS")
    lines.append("-" * 70)
    lines.append(f"Count : {len(disagreements)}")
    if disagreements:
        lines.append("Disagreeing IDs:")
        for d in disagreements:
            lines.append(f"  {d['id']}  original={d['orig_label']}  relabel={d['new_label']}")
    else:
        lines.append("Disagreeing IDs : None")
    lines.append("")
    lines.append("5. TIMING")
    lines.append("-" * 70)
    if med_sec is not None:
        lines.append(f"Median seconds spent : {med_sec:.2f} s")
    else:
        lines.append("Median seconds spent : N/A")
    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


def score_relabel(relabel_path: str = os.path.join("data", "relabel_labels.jsonl"),
                  negative_path: str = os.path.join("data", "negative_labels.jsonl"),
                  output_path: str = os.path.join("reports", "relabel_agreement.txt"),
                  expected_count: int = 20):
    relabel_items = load_relabel_labels(relabel_path)
    if expected_count is not None and len(relabel_items) != expected_count:
        sys.exit(
            f"Error: Line count mismatch in '{relabel_path}'. "
            f"Expected {expected_count}, but found {len(relabel_items)}."
        )

    orig_by_id = load_negative_labels(negative_path)

    missing_ids = []
    for r in relabel_items:
        rid = str(r.get("id"))
        if rid not in orig_by_id:
            missing_ids.append(rid)

    if missing_ids:
        sys.exit(
            f"Error: Missing active original labels for id(s): {missing_ids}"
        )

    n = len(relabel_items)
    pairs = []
    for r in relabel_items:
        rid = str(r["id"])
        new_lbl = r.get("label")
        orig_lbl = orig_by_id[rid].get("label")
        pairs.append({
            "id": rid,
            "new_label": new_lbl,
            "orig_label": orig_lbl,
            "seconds_spent": r.get("seconds_spent"),
        })

    agree_count = sum(1 for p in pairs if p["new_label"] == p["orig_label"])
    agree_pct = (agree_count / n) * 100.0 if n > 0 else 0.0
    ci_low, ci_high = wilson_interval(agree_count, n, z=1.96) if n > 0 else (None, None)
    disagreements = [p for p in pairs if p["new_label"] != p["orig_label"]]

    orig_sample_labels = [orig_by_id[str(r["id"])].get("label") for r in relabel_items]
    orig_counts = Counter(orig_sample_labels)

    seconds_list = [
        r["seconds_spent"]
        for r in relabel_items
        if "seconds_spent" in r and r["seconds_spent"] is not None
    ]
    med_sec = statistics.median(seconds_list) if seconds_list else None

    report = format_report(
        n=n,
        agree_count=agree_count,
        agree_pct=agree_pct,
        ci_low=ci_low,
        ci_high=ci_high,
        disagreements=disagreements,
        orig_counts=orig_counts,
        med_sec=med_sec,
        relabel_path=relabel_path,
        negative_path=negative_path,
        expected_count=expected_count,
    )

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report, agree_count, n


def main():
    parser = argparse.ArgumentParser(description="Score intra-annotator re-label agreement.")
    parser.add_argument(
        "--relabel",
        default=os.path.join("data", "relabel_labels.jsonl"),
        help="Path to relabel labels jsonl",
    )
    parser.add_argument(
        "--negative_labels",
        default=os.path.join("data", "negative_labels.jsonl"),
        help="Path to active negative labels jsonl",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("reports", "relabel_agreement.txt"),
        help="Path to output report",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=20,
        help="Expected line count in relabel file (default: 20)",
    )
    args = parser.parse_args()

    report, agree_count, n = score_relabel(
        relabel_path=args.relabel,
        negative_path=args.negative_labels,
        output_path=args.output,
        expected_count=args.expected_count,
    )
    print(report)


if __name__ == "__main__":
    main()

