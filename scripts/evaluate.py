import argparse
from datetime import datetime
import json
import os
import sys

from metrics import (
    accuracy, auroc, balanced_accuracy, cluster_bootstrap, confusion,
    f1, f1_compatible, macro_f1, mcc, paired_bootstrap_diff, precision, recall, specificity,
    wilson_interval
)


def fmt(v):
    return "undefined" if v is None else f"{v:.4f}"


def fmt_ci(ci):
    return "undefined" if (ci is None or ci[0] is None) else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def main():
    p = argparse.ArgumentParser(description="Evaluate conflict predictions.")
    p.add_argument("--gold", required=True, help="Gold JSONL file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preds", help="Predictions JSONL file")
    g.add_argument("--baseline", choices=["always_conflict", "always_compatible"])
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--name", required=True, help="Run name")
    p.add_argument("--max_pair_tokens", type=int, default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--compare", default=None)
    args = p.parse_args()

    if args.gold.lower().endswith(".csv"):
        import csv
        with open(args.gold, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            split_items = []
            for row in reader:
                if row.get("split") == args.split:
                    split_items.append({
                        "id": row.get("pair_id") or row.get("id"),
                        "split": row.get("split"),
                        "label": int(row["label"]),
                        "article": row["article"],
                    })
    else:
        with open(args.gold, "r", encoding="utf-8") as f:
            split_items = [x for x in (json.loads(l) for l in f if l.strip()) if x.get("split") == args.split]

    dropped = 0
    if args.max_pair_tokens is not None:
        kept = [x for x in split_items if x.get("pair_tokens") is not None and x["pair_tokens"] <= args.max_pair_tokens]
        dropped = len(split_items) - len(kept)
        split_items = kept

    items = []
    if args.preds:
        with open(args.preds, "r", encoding="utf-8") as f:
            preds_map = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}
        missing = [x["id"] for x in split_items if x["id"] not in preds_map]
        if missing:
            sys.stderr.write(f"Error: Missing predictions for {len(missing)} ids: {missing}\n")
            sys.exit(1)
        for x in split_items:
            it = dict(x, pred=preds_map[x["id"]]["pred"])
            if "score" in preds_map[x["id"]]:
                it["score"] = preds_map[x["id"]]["score"]
            items.append(it)
    else:
        b_pred = 1 if args.baseline == "always_conflict" else 0
        items = [dict(x, pred=b_pred) for x in split_items]

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            cmp_map = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}
        missing_cmp = [x["id"] for x in items if x["id"] not in cmp_map]
        if missing_cmp:
            sys.stderr.write(f"Error: Missing comparison predictions for {len(missing_cmp)} ids: {missing_cmp}\n")
            sys.exit(1)
        for it in items:
            it["other_pred"] = cmp_map[it["id"]]["pred"]

    lbls = [x["label"] for x in items]
    prds = [x["pred"] for x in items]
    tp, fp, fn, tn = confusion(lbls, prds)
    pos = sum(1 for l in lbls if l == 1)
    neg = len(lbls) - pos
    distinct_arts = len(set(x["article"] for x in items))

    prec_w = wilson_interval(tp, tp + fp)
    rec_w = wilson_interval(tp, tp + fn)
    spec_w = wilson_interval(tn, tn + fp)
    wilson_intervals = {
        "Precision": prec_w,
        "Recall": rec_w,
        "Specificity": spec_w,
    }

    metric_defs = [
        ("Precision", precision(tp=tp, fp=fp), lambda s: precision([x["label"] for x in s], [x["pred"] for x in s])),
        ("Recall", recall(tp=tp, fn=fn), lambda s: recall([x["label"] for x in s], [x["pred"] for x in s])),
        ("Specificity", specificity(tn=tn, fp=fp), lambda s: specificity([x["label"] for x in s], [x["pred"] for x in s])),
        ("F1 (conflict)", f1(tp=tp, fp=fp, fn=fn), lambda s: f1([x["label"] for x in s], [x["pred"] for x in s])),
        ("F1 (compatible)", f1_compatible(tn=tn, fp=fp, fn=fn), lambda s: f1_compatible([x["label"] for x in s], [x["pred"] for x in s])),
        ("Macro-F1", macro_f1(tp=tp, fp=fp, fn=fn, tn=tn), lambda s: macro_f1([x["label"] for x in s], [x["pred"] for x in s])),
        ("Accuracy", accuracy(tp=tp, fp=fp, fn=fn, tn=tn), lambda s: accuracy([x["label"] for x in s], [x["pred"] for x in s])),
        ("Balanced Accuracy", balanced_accuracy(tp=tp, fp=fp, fn=fn, tn=tn), lambda s: balanced_accuracy([x["label"] for x in s], [x["pred"] for x in s])),
        ("MCC", mcc(tp=tp, fp=fp, fn=fn, tn=tn), lambda s: mcc([x["label"] for x in s], [x["pred"] for x in s])),
    ]
    if all("score" in x for x in items) and len(items) > 0:
        metric_defs.append(("AUROC", auroc(lbls, [x["score"] for x in items]), lambda s: auroc([x["label"] for x in s], [x["score"] for x in s])))

    metrics_res = [(name, val, cluster_bootstrap(items, fn)[0]) for name, val, fn in metric_defs]

    lines = [
        f"Arguments: {vars(args)}", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Positives: {pos}, Negatives: {neg}, Distinct articles: {distinct_arts}",
        f"Items dropped by --max_pair_tokens: {dropped}", "",
        "Confusion Matrix:", f"  TP: {tp:<6} FP: {fp:<6}", f"  FN: {fn:<6} TN: {tn:<6}", "",
        f"{'Metric':<20} {'Value':<10} {'95% CI':<20} {'95% Wilson':<20}", "-" * 72,
    ]
    for name, val, ci in metrics_res:
        w_ci = wilson_intervals.get(name)
        w_str = fmt_ci(w_ci) if w_ci is not None else ""
        lines.append(f"{name:<20} {fmt(val):<10} {fmt_ci(ci):<20} {w_str:<20}")

    group_data = {}
    if args.group:
        lines.extend(["", f"Recall by group on positives ({args.group}):", f"{'Group':<20} {'Count':<8} {'Recall':<10}", "-" * 40])
        pos_items = [x for x in items if x["label"] == 1]
        for g_val in sorted(list({str(x.get(args.group, 'unknown')) for x in pos_items})):
            g_items = [x for x in pos_items if str(x.get(args.group, 'unknown')) == g_val]
            rec = sum(1 for x in g_items if x["pred"] == 1) / len(g_items) if g_items else None
            group_data[g_val] = {"count": len(g_items), "recall": rec}
            lines.append(f"{g_val:<20} {len(g_items):<8} {fmt(rec):<10}")

    comp_data = {}
    if args.compare:
        d_mcc, ci_mcc, _ = paired_bootstrap_diff(items, lambda s: mcc([x["label"] for x in s], [x["pred"] for x in s]), lambda s: mcc([x["label"] for x in s], [x["other_pred"] for x in s]))
        d_bacc, ci_bacc, _ = paired_bootstrap_diff(items, lambda s: balanced_accuracy([x["label"] for x in s], [x["pred"] for x in s]), lambda s: balanced_accuracy([x["label"] for x in s], [x["other_pred"] for x in s]))
        comp_data = {"MCC": {"diff": d_mcc, "ci_95": ci_mcc}, "Balanced Accuracy": {"diff": d_bacc, "ci_95": ci_bacc}}
        lines.extend(["", f"Comparison vs {args.compare} (System - Other):", f"{'Metric':<20} {'Diff':<10} {'95% CI':<20}", "-" * 52,
                      f"{'MCC':<20} {fmt(d_mcc):<10} {fmt_ci(ci_mcc):<20}", f"{'Balanced Accuracy':<20} {fmt(d_bacc):<10} {fmt_ci(ci_bacc):<20}"])

    report_txt = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("results", args.name), exist_ok=True)
    with open(os.path.join("reports", f"{args.name}_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")
    metrics_json = {}
    for name, val, ci in metrics_res:
        m_entry = {
            "value": val,
            "ci_95": list(ci) if (ci and ci[0] is not None) else None,
        }
        if name in wilson_intervals:
            w_lo, w_hi = wilson_intervals[name]
            m_entry["wilson_lo"] = w_lo
            m_entry["wilson_hi"] = w_hi
            m_entry[f"{name.lower()}_wilson_lo"] = w_lo
            m_entry[f"{name.lower()}_wilson_hi"] = w_hi
        metrics_json[name] = m_entry

    json_obj = {
        "args": vars(args), "date": datetime.now().isoformat(),
        "counts": {"positives": pos, "negatives": neg, "articles": distinct_arts, "dropped_tokens": dropped},
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "metrics": metrics_json,
        "precision_wilson_lo": prec_w[0],
        "precision_wilson_hi": prec_w[1],
        "recall_wilson_lo": rec_w[0],
        "recall_wilson_hi": rec_w[1],
        "specificity_wilson_lo": spec_w[0],
        "specificity_wilson_hi": spec_w[1],
    }
    if group_data: json_obj["group"] = group_data
    if comp_data: json_obj["compare"] = comp_data
    with open(os.path.join("results", args.name, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)
    print(report_txt)


if __name__ == "__main__":
    main()
