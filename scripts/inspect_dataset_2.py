import os
import sys
from collections import Counter, defaultdict
import numpy as np

# CPU only & environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

REPORT_PATH = os.path.join("reports", "dataset_inspection_2.txt")
DATASET_ID = "ibm/Wikipedia_contradict_benchmark"
SPLIT_NAME = "train"

lines = []

def log(text=""):
    print(text)
    lines.append(text)

def truncate(val, max_len=200):
    s = str(val) if val is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."

def main():
    os.makedirs("reports", exist_ok=True)

    try:
        from datasets import load_dataset
        log(f"Loading dataset '{DATASET_ID}' (split: {SPLIT_NAME})...")
        ds = load_dataset(DATASET_ID, split=SPLIT_NAME)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    log("=" * 60)
    log("WIKICONTRADICT DATASET INSPECTION REPORT 2")
    log("=" * 60)
    log(f"Total rows in '{SPLIT_NAME}' split: {len(ds)}")

    # 1. Unique question_ID values
    log("\n1. UNIQUE QUESTION_ID VALUES")
    log("-" * 35)
    q_ids = [row["question_ID"] for row in ds]
    unique_q_ids = len(set(q_ids))
    log(f"Total question_ID entries: {len(q_ids)}")
    log(f"Unique question_ID count: {unique_q_ids}")
    if unique_q_ids == len(q_ids):
        log("All question_IDs are distinct.")
    else:
        log(f"Duplicate question_IDs exist ({len(q_ids) - unique_q_ids} duplicates).")

    # 2. Unique (context1, context2) pairs & repeated pairs
    log("\n2. UNIQUE (CONTEXT1, CONTEXT2) PAIRS AND REPETITIONS")
    log("-" * 35)
    pair_to_rows = defaultdict(list)
    for idx, row in enumerate(ds):
        pair = (row["context1"], row["context2"])
        pair_to_rows[pair].append(row)

    unique_pairs_count = len(pair_to_rows)
    repeated_pairs = {k: v for k, v in pair_to_rows.items() if len(v) > 1}
    log(f"Number of unique (context1, context2) pairs: {unique_pairs_count}")
    log(f"Number of pairs occurring more than once: {len(repeated_pairs)}")

    if repeated_pairs:
        log(f"\nDetails for up to 3 repeated pairs (showing all matching rows):")
        for i, (pair, rows) in enumerate(list(repeated_pairs.items())[:3], 1):
            log(f"\n--- Repeated Pair {i} (occurs {len(rows)} times) ---")
            log(f"  context1 preview: {truncate(pair[0], 120)}")
            log(f"  context2 preview: {truncate(pair[1], 120)}")
            for r_idx, r in enumerate(rows, 1):
                log(f"  Row instance {r_idx}:")
                log(f"    question_ID:    {r['question_ID']}")
                log(f"    question:       {truncate(r['question'], 150)}")
                log(f"    answer1:        {truncate(r['answer1'], 100)}")
                log(f"    answer2:        {truncate(r['answer2'], 100)}")
                log(f"    contradictType: {r['contradictType']}")

    # 3. Distribution of samepassage
    log("\n3. DISTRIBUTION OF SAMEPASSAGE")
    log("-" * 35)
    sp_counter = Counter(row["samepassage"] for row in ds)
    for val, count in sp_counter.most_common():
        pct = (count / len(ds)) * 100
        log(f"  - '{val}': {count} ({pct:.2f}%)")

    # 4. WikipediaArticleTitle statistics
    log("\n4. WIKIPEDIA ARTICLE TITLE STATISTICS")
    log("-" * 35)
    article_counts = Counter(row["WikipediaArticleTitle"] for row in ds)
    unique_articles = len(article_counts)
    counts_list = list(article_counts.values())
    min_rows = int(np.min(counts_list))
    median_rows = float(np.median(counts_list))
    max_rows = int(np.max(counts_list))
    log(f"Number of unique WikipediaArticleTitle values: {unique_articles}")
    log(f"Rows per article - Min: {min_rows}, Median: {median_rows:.1f}, Max: {max_rows}")

    # 5. Cross-tab of contradictType by samepassage
    log("\n5. CROSS-TABULATION: CONTRADICTTYPE BY SAMEPASSAGE")
    log("-" * 35)
    c_types = sorted(list(set(row["contradictType"] for row in ds)))
    s_types = sorted(list(set(row["samepassage"] for row in ds)))

    grid = {ct: Counter() for ct in c_types}
    for row in ds:
        grid[row["contradictType"]][row["samepassage"]] += 1

    # Formatting table
    header = f"{'contradictType / samepassage':<32} | " + " | ".join(f"{st:>12}" for st in s_types) + " | " + f"{'Total':>8}"
    sep = "-" * len(header)
    log(header)
    log(sep)
    for ct in c_types:
        row_str = f"{ct:<32} | "
        row_counts = [grid[ct][st] for st in s_types]
        row_total = sum(row_counts)
        row_str += " | ".join(f"{c:>12}" for c in row_counts)
        row_str += f" | {row_total:>8}"
        log(row_str)
    log(sep)
    col_totals = [sum(grid[ct][st] for ct in c_types) for st in s_types]
    total_str = f"{'Total':<32} | " + " | ".join(f"{c:>12}" for c in col_totals) + f" | {len(ds):>8}"
    log(total_str)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nReport successfully saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()

