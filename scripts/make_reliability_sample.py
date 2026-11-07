"""
Generate reliability sample files for intra-annotator and inter-annotator (co-author) reliability.

Spec 17 requirements:
1. Intra-annotator re-label set: 20 random COMPATIBLE negatives from gold test negatives,
   seed 42, ids sorted before sampling. Write data/relabel_sample.jsonl with fields
   id, passage1, passage2 only (no gold label, no model output), shuffled.
2. Co-author set: 25 test negatives and 20 test positives, disjoint from step 1's 20 where possible,
   seed 42, sorted ids first. Write data/coauthor_sample.jsonl with id, passage1, passage2 only,
   shuffled, with neutral id (c001, c002, ...) and separate data/coauthor_key.jsonl
   mapping neutral id to gold id and label.
3. Co-author sheet: data/coauthor_sheet.csv with columns neutral id, passage1, passage2, label (empty).
4. Report: reports/reliability_sample_plan.txt with sizes, seed, overlap, and SHA-256 hashes.
   No passage text in report.
"""

import os
import sys
import json
import csv
import hashlib
import random


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    gold_path = os.path.join("data", "gold.jsonl")
    if not os.path.exists(gold_path):
        print(f"Error: {gold_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]

    # Filter test split
    test_negs = [x for x in gold if x.get("split") == "test" and x.get("label") == 0]
    test_pos = [x for x in gold if x.get("split") == "test" and x.get("label") == 1]

    # --- Step 1: Intra-annotator re-label set ---
    # 20 random COMPATIBLE negatives from gold test negatives, seed 42, ids sorted before sampling.
    sorted_test_negs = sorted(test_negs, key=lambda x: x["id"])
    rng_step1 = random.Random(42)
    relabel_sample = rng_step1.sample(sorted_test_negs, 20)

    os.makedirs("data", exist_ok=True)
    relabel_path = os.path.join("data", "relabel_sample.jsonl")
    with open(relabel_path, "w", encoding="utf-8") as f:
        for x in relabel_sample:
            row = {
                "id": x["id"],
                "passage1": x["passage1"],
                "passage2": x["passage2"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Step 2: Co-author set ---
    # 25 test negatives and 20 test positives, disjoint from step 1's 20 where possible, seed 42, sorted ids first.
    relabel_ids = {x["id"] for x in relabel_sample}
    rem_negs = [x for x in test_negs if x["id"] not in relabel_ids]
    sorted_rem_negs = sorted(rem_negs, key=lambda x: x["id"])
    sorted_test_pos = sorted(test_pos, key=lambda x: x["id"])

    rng_step2 = random.Random(42)
    coauthor_negs = rng_step2.sample(sorted_rem_negs, 25)
    coauthor_pos = rng_step2.sample(sorted_test_pos, 20)

    coauthor_combined = coauthor_negs + coauthor_pos
    rng_step2.shuffle(coauthor_combined)

    # Assign neutral IDs c001, c002, ...
    coauthor_sample_path = os.path.join("data", "coauthor_sample.jsonl")
    coauthor_key_path = os.path.join("data", "coauthor_key.jsonl")

    key_records = []
    with open(coauthor_sample_path, "w", encoding="utf-8") as f_samp, \
         open(coauthor_key_path, "w", encoding="utf-8") as f_key:
        for idx, x in enumerate(coauthor_combined, start=1):
            neutral_id = f"c{idx:03d}"
            samp_row = {
                "id": neutral_id,
                "passage1": x["passage1"],
                "passage2": x["passage2"],
            }
            f_samp.write(json.dumps(samp_row, ensure_ascii=False) + "\n")

            label_name = "CONTRADICT" if x["label"] == 1 else "COMPATIBLE"
            key_row = {
                "neutral_id": neutral_id,
                "id": neutral_id,
                "gold_id": x["id"],
                "label": x["label"],
                "gold_label": label_name,
            }
            key_records.append(key_row)
            f_key.write(json.dumps(key_row, ensure_ascii=False) + "\n")

    # --- Step 3: Co-author sheet CSV ---
    # columns neutral id, passage1, passage2, label (empty)
    sheet_path = os.path.join("data", "coauthor_sheet.csv")
    with open(sheet_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["neutral id", "passage1", "passage2", "label"])
        for idx, x in enumerate(coauthor_combined, start=1):
            neutral_id = f"c{idx:03d}"
            writer.writerow([neutral_id, x["passage1"], x["passage2"], ""])

    # --- Step 4: Plan report ---
    overlap_ids = sorted(list(relabel_ids & {x["id"] for x in coauthor_combined}))

    relabel_hash = compute_sha256(relabel_path)
    coauthor_sample_hash = compute_sha256(coauthor_sample_path)
    coauthor_key_hash = compute_sha256(coauthor_key_path)
    sheet_hash = compute_sha256(sheet_path)

    os.makedirs("reports", exist_ok=True)
    plan_path = os.path.join("reports", "reliability_sample_plan.txt")

    lines = [
        "Reliability Sampling Plan (Spec 17)",
        "====================================",
        "",
        "Sampling Parameters:",
        "  Random seed: 42",
        "  Gold test pool size: 151 (73 negatives, 78 positives)",
        "",
        "Sample Sizes:",
        f"  Intra-annotator re-label set (data/relabel_sample.jsonl): {len(relabel_sample)} items",
        f"    - Subsets: 20 COMPATIBLE negatives",
        f"  Co-author set (data/coauthor_sample.jsonl): {len(coauthor_combined)} items",
        f"    - Subsets: {len(coauthor_negs)} negatives, {len(coauthor_pos)} positives",
        f"  Co-author key (data/coauthor_key.jsonl): {len(key_records)} items",
        f"  Co-author sheet (data/coauthor_sheet.csv): {len(coauthor_combined)} rows",
        "",
        "Overlap Between Sets:",
        f"  Overlap count: {len(overlap_ids)} items",
        f"  Disjoint: {len(overlap_ids) == 0}",
        "",
        "File SHA-256 Checksums:",
        f"  data/relabel_sample.jsonl:  {relabel_hash}",
        f"  data/coauthor_sample.jsonl: {coauthor_sample_hash}",
        f"  data/coauthor_key.jsonl:    {coauthor_key_hash}",
        f"  data/coauthor_sheet.csv:    {sheet_hash}",
    ]

    report_content = "\n".join(lines) + "\n"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)


if __name__ == "__main__":
    main()
