"""
Spec 23 — Co-author scoring script (reproducible, no model runs).

Reads:
  data/gold.jsonl              — master gold file (read-only, hash-checked)
  data/coauthor_key.jsonl      — neutral-id → gold_id / gold_label mapping
  data/coauthor_sheet_result.csv — co-author labels (neutral id, label)

Writes:
  reports/coauthor_agreement.txt

Reproduces the report committed under Spec 21 exactly.
"""

import csv
import hashlib
import json
import os
import random
import sys


GOLD_PATH = os.path.join("data", "gold.jsonl")
KEY_PATH = os.path.join("data", "coauthor_key.jsonl")
RESULT_PATH = os.path.join("data", "coauthor_sheet_result.csv")
REPORT_PATH = os.path.join("reports", "coauthor_agreement.txt")

VALID_LABELS = {"COMPATIBLE", "CONTRADICT", "UNSURE"}
EXPECTED_ROWS = 45

BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def cohen_kappa(pairs, label_set):
    """Compute Cohen's kappa for a list of (gold, coauthor) string pairs."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels = sorted(label_set)

    # Build confusion matrix
    conf = {g: {c: 0 for c in labels} for g in labels}
    for g, c in pairs:
        conf[g][c] += 1

    # Observed agreement
    p_o = sum(conf[l][l] for l in labels) / n

    # Expected agreement
    p_e = 0.0
    for l in labels:
        row_sum = sum(conf[l][c] for c in labels)
        col_sum = sum(conf[g][l] for g in labels)
        p_e += (row_sum / n) * (col_sum / n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_kappa_ci(pairs, label_set, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                        alpha=0.05):
    """Return (lower, upper) 95% bootstrap CI for Cohen's kappa."""
    rng = random.Random(seed)
    n_pairs = len(pairs)
    kappas = []
    for _ in range(n):
        sample = [pairs[rng.randint(0, n_pairs - 1)] for _ in range(n_pairs)]
        kappas.append(cohen_kappa(sample, label_set))
    kappas.sort()
    lo_idx = int(alpha / 2 * n)
    hi_idx = int((1 - alpha / 2) * n) - 1
    return kappas[lo_idx], kappas[hi_idx]


def confusion_table(pairs, all_labels, gold_label_filter=None):
    """
    Build a confusion table dict.
    pairs: list of (gold, coauthor)
    gold_label_filter: if set, only include rows where gold == gold_label_filter
    """
    cols = ["COMPATIBLE", "CONTRADICT", "UNSURE"]
    if gold_label_filter is not None:
        rows_labels = [gold_label_filter]
        filtered = [(g, c) for g, c in pairs if g == gold_label_filter]
    else:
        rows_labels = ["COMPATIBLE", "CONTRADICT"]
        filtered = pairs

    conf = {g: {c: 0 for c in cols} for g in rows_labels}
    for g, c in filtered:
        if g in conf:
            conf[g][c] += 1
    return conf, filtered, rows_labels, cols


LABEL_W = 16        # row label left-aligned width for data rows
COL_WIDTHS = [10, 12, 12, 8]   # COMPATIBLE, CONTRADICT, UNSURE, TOTAL


def render_confusion_table(conf, rows_labels, cols):
    """Render a confusion table as a string block matching the committed format.

    Header label is left-padded to LABEL_W+1; data row labels to LABEL_W.
    Data columns are right-aligned at COL_WIDTHS.
    Separator length equals the header length.
    """
    # Header (label_w+1 to leave room for the extra space after 'coauthor')
    header = "gold \\ coauthor".ljust(LABEL_W + 1)
    for col, w in zip(cols + ["TOTAL"], COL_WIDTHS):
        header += col.rjust(w)
    sep = "-" * len(header)
    lines = [header, sep]

    col_totals = {c: 0 for c in cols}
    for g in rows_labels:
        row_total = sum(conf[g][c] for c in cols)
        row = g.ljust(LABEL_W)
        for col, w in zip(cols, COL_WIDTHS[:-1]):
            row += str(conf[g][col]).rjust(w)
            col_totals[col] += conf[g][col]
        row += str(row_total).rjust(COL_WIDTHS[-1])
        lines.append(row)

    lines.append(sep)
    grand_total = sum(col_totals.values())
    total_row = "TOTAL".ljust(LABEL_W)
    for col, w in zip(cols, COL_WIDTHS[:-1]):
        total_row += str(col_totals[col]).rjust(w)
    total_row += str(grand_total).rjust(COL_WIDTHS[-1])
    lines.append(total_row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Gold hash BEFORE ---
    if not os.path.exists(GOLD_PATH):
        print(f"Error: {GOLD_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    gold_hash_before = sha256_file(GOLD_PATH)

    # --- Load key ---
    if not os.path.exists(KEY_PATH):
        print(f"Error: {KEY_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    key = {}  # neutral_id → gold_label
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key[r["neutral_id"]] = r["gold_label"]

    # --- Load co-author sheet result ---
    if not os.path.exists(RESULT_PATH):
        print(f"Error: {RESULT_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    coauthor = {}  # neutral_id → label
    errors = []
    with open(RESULT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = row["neutral id"].strip()
            lbl = row["label"].strip()
            coauthor[nid] = lbl

    # --- Sheet validation ---
    row_count = len(coauthor)
    valid_label_count = sum(1 for lbl in coauthor.values() if lbl in VALID_LABELS)

    for nid, lbl in sorted(coauthor.items()):
        if nid not in key:
            errors.append(f"  Row {nid}: id not found in key")
        if lbl not in VALID_LABELS:
            errors.append(f"  Row {nid}: invalid label '{lbl}'")

    # --- Build pairs (sorted by neutral_id) ---
    all_ids = sorted(key.keys())
    pairs_all = [(key[nid], coauthor[nid]) for nid in all_ids]

    # --- Confusion tables ---
    all_labels = {"COMPATIBLE", "CONTRADICT", "UNSURE"}

    # All 45 rows
    conf_all, _, rows_all, cols = confusion_table(pairs_all, all_labels)

    # Negatives only (gold=COMPATIBLE)
    conf_neg, pairs_neg, rows_neg, _ = confusion_table(pairs_all, all_labels, "COMPATIBLE")

    # Positives only (gold=CONTRADICT)
    conf_pos, pairs_pos, rows_pos, _ = confusion_table(pairs_all, all_labels, "CONTRADICT")

    # --- Agreement statistics ---
    # (a) All three labels
    n_all = len(pairs_all)
    agree_all = sum(1 for g, c in pairs_all if g == c)
    pct_all = agree_all / n_all * 100
    kappa_all = cohen_kappa(pairs_all, all_labels)
    ci_all_lo, ci_all_hi = bootstrap_kappa_ci(pairs_all, all_labels)

    # (b) UNSURE rows dropped
    pairs_no_unsure = [(g, c) for g, c in pairs_all if c != "UNSURE"]
    n_dropped = n_all - len(pairs_no_unsure)
    n_no_unsure = len(pairs_no_unsure)
    agree_no_unsure = sum(1 for g, c in pairs_no_unsure if g == c)
    pct_no_unsure = agree_no_unsure / n_no_unsure * 100
    labels_no_unsure = {"COMPATIBLE", "CONTRADICT"}
    kappa_no_unsure = cohen_kappa(pairs_no_unsure, labels_no_unsure)
    ci_nu_lo, ci_nu_hi = bootstrap_kappa_ci(pairs_no_unsure, labels_no_unsure)

    # --- Disagreement IDs ---
    disagreements = [
        (nid, key[nid], coauthor[nid])
        for nid in all_ids
        if key[nid] != coauthor[nid]
    ]

    # --- Gold hash AFTER (file not modified) ---
    gold_hash_after = sha256_file(GOLD_PATH)
    hash_unchanged = gold_hash_before == gold_hash_after

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    sep_eq = "=" * 70
    sep_dashes = "-" * 70
    sep_short = "-" * 42

    lines = [
        sep_eq,
        "SPEC 21 — Co-author Sheet Agreement Report",
        sep_eq,
        "",
        f"gold.jsonl SHA-256 BEFORE : {gold_hash_before}",
        "",
        "1. SHEET VALIDATION",
        "-" * 40,
        f"   Row count : {row_count} (expected {EXPECTED_ROWS}) — {'OK' if row_count == EXPECTED_ROWS else 'FAIL'}",
        f"   Valid labels: {valid_label_count}/{row_count}",
    ]
    if errors:
        lines.append("   Validation errors:")
        lines.extend(errors)
    else:
        lines.append("   No validation errors — all ids match key, all labels valid.")

    lines += [
        "",
        "2. CONFUSION TABLES  (rows = gold, cols = co-author)",
        sep_dashes,
        "",
        f"ALL {row_count} ROWS",
        render_confusion_table(conf_all, rows_all, cols),
        "",
        f"NEGATIVES ONLY ({len(pairs_neg)} rows, gold=COMPATIBLE)",
        render_confusion_table(conf_neg, rows_neg, cols),
        "",
        f"POSITIVES ONLY ({len(pairs_pos)} rows, gold=CONTRADICT)",
        render_confusion_table(conf_pos, rows_pos, cols),
        "",
        "3. AGREEMENT STATISTICS",
        sep_dashes,
        "",
        "  (a) All three labels as given:",
        f"      N                   : {n_all}",
        f"      % agreement         : {pct_all:.1f}%",
        f"      Cohen's kappa       : {kappa_all:.4f}",
        f"      95% bootstrap CI    : [{ci_all_lo:.4f}, {ci_all_hi:.4f}]  (seed={BOOTSTRAP_SEED}, n={BOOTSTRAP_N})",
        "",
        f"  (b) UNSURE rows dropped ({n_dropped} dropped, leaving {n_no_unsure}):",
        f"      N                   : {n_no_unsure}",
        f"      % agreement         : {pct_no_unsure:.1f}%",
        f"      Cohen's kappa       : {kappa_no_unsure:.4f}",
        f"      95% bootstrap CI    : [{ci_nu_lo:.4f}, {ci_nu_hi:.4f}]  (seed={BOOTSTRAP_SEED}, n={BOOTSTRAP_N})",
        "",
        "4. DISAGREEMENT IDs",
        sep_dashes,
        f"   Count: {len(disagreements)}",
    ]
    for nid, gold_lbl, ca_lbl in disagreements:
        lines.append(f"   {nid}  gold={gold_lbl}  coauthor={ca_lbl}")

    lines += [
        "",
        f"gold.jsonl SHA-256 AFTER  : {gold_hash_after}",
        f"Hash unchanged            : {hash_unchanged}",
    ]

    report = "\n".join(lines)

    os.makedirs("reports", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()
