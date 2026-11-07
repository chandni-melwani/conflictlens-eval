import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.metrics import (
    auroc,
    balanced_accuracy,
    mcc,
    paired_bootstrap_diff,
)


def fmt(v):
    return "undefined" if v is None else f"{v:.4f}"


def fmt_ci(ci):
    return "undefined" if (ci is None or ci[0] is None) else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def paired_auroc_bootstrap(labels, scores_a, scores_b, n=2000, seed=42):
    """
    Paired bootstrap over items, resampling items jointly with replacement.
    Evaluates AUROC(scores_a) - AUROC(scores_b).
    Returns (val_a, val_b, diff, ci_95, skipped).
    """
    labels = list(labels)
    scores_a = list(scores_a)
    scores_b = list(scores_b)
    val_a = auroc(labels, scores_a)
    val_b = auroc(labels, scores_b)
    diff = (val_a - val_b) if (val_a is not None and val_b is not None) else None

    n_items = len(labels)
    if n_items == 0:
        return val_a, val_b, diff, (None, None), n

    rng = np.random.default_rng(seed)
    diffs = []
    skipped = 0
    for _ in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        s_lbl = [labels[i] for i in idx]
        if not (any(l == 1 for l in s_lbl) and any(l == 0 for l in s_lbl)):
            skipped += 1
            continue
        sa = auroc(s_lbl, [scores_a[i] for i in idx])
        sb = auroc(s_lbl, [scores_b[i] for i in idx])
        if sa is None or sb is None:
            skipped += 1
            continue
        diffs.append(sa - sb)

    if not diffs:
        return val_a, val_b, diff, (None, None), skipped
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return val_a, val_b, diff, ci, skipped


def run_check_a(gold_items, judge_map, nli_map, max_pair_tokens=None, n=2000, seed=42):
    items = gold_items
    if max_pair_tokens is not None:
        items = [x for x in items if x.get("pair_tokens") is not None and x["pair_tokens"] <= max_pair_tokens]

    merged = []
    for x in items:
        it = dict(x, pred=judge_map[x["id"]]["pred"], other_pred=nli_map[x["id"]]["pred"])
        merged.append(it)

    m_judge = mcc([x["label"] for x in merged], [x["pred"] for x in merged])
    b_judge = balanced_accuracy([x["label"] for x in merged], [x["pred"] for x in merged])
    m_nli = mcc([x["label"] for x in merged], [x["other_pred"] for x in merged])
    b_nli = balanced_accuracy([x["label"] for x in merged], [x["other_pred"] for x in merged])

    d_mcc, ci_mcc, _ = paired_bootstrap_diff(
        merged,
        lambda s: mcc([x["label"] for x in s], [x["pred"] for x in s]),
        lambda s: mcc([x["label"] for x in s], [x["other_pred"] for x in s]),
        n=n, seed=seed
    )
    d_bacc, ci_bacc, _ = paired_bootstrap_diff(
        merged,
        lambda s: balanced_accuracy([x["label"] for x in s], [x["pred"] for x in s]),
        lambda s: balanced_accuracy([x["label"] for x in s], [x["other_pred"] for x in s]),
        n=n, seed=seed
    )

    return {
        "n_items": len(merged),
        "positives": sum(1 for x in merged if x["label"] == 1),
        "negatives": sum(1 for x in merged if x["label"] == 0),
        "judge_mcc": m_judge,
        "judge_bacc": b_judge,
        "nli_mcc": m_nli,
        "nli_bacc": b_nli,
        "diff_mcc": d_mcc,
        "ci_mcc": ci_mcc,
        "diff_bacc": d_bacc,
        "ci_bacc": ci_bacc,
    }


def find_saved_scores_file(candidates: list[str]) -> str | None:
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Spec 12: Paired checks on saved test predictions")
    parser.add_argument("--gold", default=os.path.join("data", "gold.jsonl"))
    parser.add_argument("--judge_preds", default=os.path.join("results", "llm_judge_test", "preds.jsonl"))
    parser.add_argument("--nli_preds", default=os.path.join("results", "nli_only_test", "preds.jsonl"))
    parser.add_argument("--style_preds", default=None, help="Explicit path to saved style test scores if available")
    parser.add_argument("--output", default=os.path.join("reports", "paired_checks_test.txt"))
    args = parser.parse_args()

    # Load gold test items
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_test = [json.loads(line) for line in f if line.strip() and json.loads(line).get("split") == "test"]

    # Load Check A prediction files
    with open(args.judge_preds, "r", encoding="utf-8") as f:
        judge_map = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}
    with open(args.nli_preds, "r", encoding="utf-8") as f:
        nli_map = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}

    res_a_full = run_check_a(gold_test, judge_map, nli_map, max_pair_tokens=None)
    res_a_120 = run_check_a(gold_test, judge_map, nli_map, max_pair_tokens=120)

    # Check B: verify saved test scores
    missing_scores = []
    # Check NLI-only test scores
    nli_has_scores = all("score" in nli_map.get(it["id"], {}) for it in gold_test)
    if not nli_has_scores:
        missing_scores.append("NLI-only")

    # Check metadata-only classifier test scores
    style_preds_candidates = [
        args.style_preds,
        os.path.join("results", "style_only_test", "preds.jsonl"),
        os.path.join("results", "style_test", "preds.jsonl"),
        os.path.join("results", "style_classifier_test", "preds.jsonl"),
        os.path.join("results", "style_checks", "preds.jsonl"),
        os.path.join("results", "metadata_only_test", "preds.jsonl"),
        os.path.join("data", "style_preds_test.jsonl"),
    ]
    style_preds_file = find_saved_scores_file([p for p in style_preds_candidates if p])
    style_scores = None
    if style_preds_file is None:
        missing_scores.append("metadata-only classifier")
    else:
        with open(style_preds_file, "r", encoding="utf-8") as f:
            style_map = {row["id"]: row for row in (json.loads(l) for l in f if l.strip())}
        if not all("score" in style_map.get(it["id"], {}) for it in gold_test):
            missing_scores.append("metadata-only classifier")
        else:
            style_scores = [style_map[it["id"]]["score"] for it in gold_test]

    labels = [it["label"] for it in gold_test]
    nli_scores = [nli_map[it["id"]]["score"] for it in gold_test]

    # Diagnostic calculation if style scores were not saved
    diagnostic_style_auroc = None
    diagnostic_diff = None
    diagnostic_ci_item = None
    diagnostic_ci_cluster = None
    try:
        from scripts.style_checks import extract_features, fit_logistic_regression, predict_proba
        with open(args.gold, "r", encoding="utf-8") as f:
            all_items = [json.loads(l) for l in f if l.strip()]
        dev_items = [it for it in all_items if it.get("split") == "dev"]
        X_dev, y_dev = extract_features(dev_items)
        X_test, _ = extract_features(gold_test)
        w, mu, sigma = fit_logistic_regression(X_dev, y_dev)
        diag_scores = predict_proba(X_test, w, mu, sigma)
        val_style, val_nli, d_diag, ci_item, _ = paired_auroc_bootstrap(labels, diag_scores, nli_scores, n=2000, seed=42)
        diagnostic_style_auroc = val_style
        diagnostic_diff = d_diag
        diagnostic_ci_item = ci_item

        # Clustered version
        diag_merged = [dict(it, score_style=diag_scores[i], score_nli=nli_scores[i]) for i, it in enumerate(gold_test)]
        _, ci_cluster, _ = paired_bootstrap_diff(
            diag_merged,
            lambda s: auroc([x["label"] for x in s], [x["score_style"] for x in s]),
            lambda s: auroc([x["label"] for x in s], [x["score_nli"] for x in s]),
            n=2000, seed=42
        )
        diagnostic_ci_cluster = ci_cluster
    except Exception as e:
        sys.stderr.write(f"Diagnostic computation error: {e}\n")

    style_display = style_preds_file.replace("\\", "/") if style_preds_file is not None else "NOT SAVED"

    lines = [
        "=" * 70,
        "SPEC 12: PAIRED CHECKS ON TEST PREDICTIONS (151 ITEMS)",
        "=" * 70,
        "",
        "CHECK A: LLM JUDGE VS NLI-ONLY",
        "  System: LLM Judge (results/llm_judge_test/preds.jsonl)",
        "  Other:  NLI-only (results/nli_only_test/preds.jsonl)",
        "  Method: Paired cluster bootstrap by article (evaluate.py --compare), seed=42, n=2000",
        "",
        f"--- Full Test Set ({res_a_full['n_items']} items: {res_a_full['positives']} positives, {res_a_full['negatives']} negatives) ---",
        f"  System (LLM Judge):  MCC = {fmt(res_a_full['judge_mcc'])}, Balanced Accuracy = {fmt(res_a_full['judge_bacc'])}",
        f"  Other (NLI-only):    MCC = {fmt(res_a_full['nli_mcc'])}, Balanced Accuracy = {fmt(res_a_full['nli_bacc'])}",
        "",
        f"  {'Metric':<20} {'Diff (System - Other)':<25} {'95% CI':<20}",
        "  " + "-" * 65,
        f"  {'MCC':<20} {fmt(res_a_full['diff_mcc']):<25} {fmt_ci(res_a_full['ci_mcc']):<20}",
        f"  {'Balanced Accuracy':<20} {fmt(res_a_full['diff_bacc']):<25} {fmt_ci(res_a_full['ci_bacc']):<20}",
        "",
        f"--- pair_tokens <= 120 ({res_a_120['n_items']} items: {res_a_120['positives']} positives, {res_a_120['negatives']} negatives) ---",
        f"  System (LLM Judge):  MCC = {fmt(res_a_120['judge_mcc'])}, Balanced Accuracy = {fmt(res_a_120['judge_bacc'])}",
        f"  Other (NLI-only):    MCC = {fmt(res_a_120['nli_mcc'])}, Balanced Accuracy = {fmt(res_a_120['nli_bacc'])}",
        "",
        f"  {'Metric':<20} {'Diff (System - Other)':<25} {'95% CI':<20}",
        "  " + "-" * 65,
        f"  {'MCC':<20} {fmt(res_a_120['diff_mcc']):<25} {fmt_ci(res_a_120['ci_mcc']):<20}",
        f"  {'Balanced Accuracy':<20} {fmt(res_a_120['diff_bacc']):<25} {fmt_ci(res_a_120['ci_bacc']):<20}",
        "",
        "=" * 70,
        "CHECK B: AUROC DIFFERENCE (METADATA-ONLY CLASSIFIER VS NLI-ONLY)",
        "=" * 70,
        "Check saved test scores:",
        f"  NLI-only:                 {'FOUND (results/nli_only_test/preds.jsonl)' if nli_has_scores else 'MISSING'}",
        f"  Metadata-only classifier: {'FOUND (' + style_display + ')' if style_scores is not None else 'NOT SAVED'}",
        "",
    ]

    if missing_scores:
        lines.extend([
            f"Status: STOPPED (Missing saved test scores for: {', '.join(missing_scores)})",
            "Rule: 'If test scores for either system are not saved, stop and report which are missing. Do not recompute silently.'",
            "",
            "[DIAGNOSTIC REFERENCE ONLY: Evaluated using dev-fitted weights without refitting]",
            f"  Metadata-only AUROC:                  {fmt(diagnostic_style_auroc)}",
            f"  NLI-only AUROC:                       {fmt(auroc(labels, nli_scores))}",
            f"  AUROC Difference (Style - NLI-only):  {fmt(diagnostic_diff)}",
            f"  95% CI (joint item bootstrap):        {fmt_ci(diagnostic_ci_item)}",
            f"  95% CI (cluster bootstrap):           {fmt_ci(diagnostic_ci_cluster)}",
            "=" * 70,
        ])
    else:
        # Full run when saved scores exist
        val_style, val_nli, d_auroc, ci_auroc, _ = paired_auroc_bootstrap(labels, style_scores, nli_scores, n=2000, seed=42)
        merged = [dict(it, score_style=style_scores[i], score_nli=nli_scores[i]) for i, it in enumerate(gold_test)]
        _, ci_cluster, _ = paired_bootstrap_diff(
            merged,
            lambda s: auroc([x["label"] for x in s], [x["score_style"] for x in s]),
            lambda s: auroc([x["label"] for x in s], [x["score_nli"] for x in s]),
            n=2000, seed=42
        )
        lines.extend([
            f"  Metadata-only AUROC:                  {fmt(val_style)}",
            f"  NLI-only AUROC:                       {fmt(val_nli)}",
            f"  AUROC Difference (Style - NLI-only):  {fmt(d_auroc)}",
            f"  95% CI (joint item bootstrap):        {fmt_ci(ci_auroc)}",
            f"  95% CI (cluster bootstrap):           {fmt_ci(ci_cluster)}",
            "=" * 70,
        ])

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(report_text)


if __name__ == "__main__":
    main()

