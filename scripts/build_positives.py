import os
import sys
import json
import re
from collections import Counter, defaultdict
import numpy as np

# Force CPU only & disable symlinks warning
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_ID = "ibm/Wikipedia_contradict_benchmark"
SPLIT_NAME = "train"
TOKENIZER_ID = "cross-encoder/nli-deberta-v3-base"
OUTPUT_JSONL_PATH = os.path.join("data", "positives.jsonl")
REPORT_PATH = os.path.join("reports", "positives_report.txt")

lines = []

def log(text=""):
    print(text)
    lines.append(text)

def clean_whitespace(s):
    if not s:
        return ""
    return " ".join(s.split())

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    log("=" * 70)
    log("BUILD POSITIVES: WIKICONTRADICT BENCHMARK")
    log("=" * 70)

    # Hub info and dataset loading
    try:
        import huggingface_hub
        api = huggingface_hub.HfApi()
        dataset_info = api.dataset_info(DATASET_ID)
        hub_sha = dataset_info.sha
        last_modified = getattr(dataset_info, "last_modified", getattr(dataset_info, "lastModified", "N/A"))
    except Exception as e:
        hub_sha = f"Error fetching sha: {e}"
        last_modified = "N/A"

    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_ID, split=SPLIT_NAME)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    cache_files = getattr(ds, "cache_files", [])

    # 1. Row counts before and after dedup across all columns
    raw_count = len(ds)
    seen_rows = set()
    deduped_rows = []
    for row in ds:
        # Convert row dict items to sorted tuple to represent full row content
        row_tuple = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in row.items()))
        if row_tuple not in seen_rows:
            seen_rows.add(row_tuple)
            deduped_rows.append(row)

    deduped_count = len(deduped_rows)

    # 2. Strip & collapse whitespace, group by (context1, context2)
    # Do not reorder the two passages
    pair_groups = defaultdict(list)
    for row in deduped_rows:
        c1 = clean_whitespace(row.get("context1", ""))
        c2 = clean_whitespace(row.get("context2", ""))
        pair_groups[(c1, c2)].append(row)

    unique_ordered_pairs_count = len(pair_groups)
    unordered_pairs = set(tuple(sorted([c1, c2])) for (c1, c2) in pair_groups.keys())
    unique_unordered_pairs_count = len(unordered_pairs)

    # Prepare intermediate records before pair_id assignment
    intermediate_records = []
    for (c1, c2), rows in pair_groups.items():
        # Article title
        titles = list(dict.fromkeys(r.get("WikipediaArticleTitle", "") for r in rows if r.get("WikipediaArticleTitle") is not None))
        # Contradict type
        c_types = list(dict.fromkeys(r.get("contradictType", "") for r in rows if r.get("contradictType") is not None))
        # Samepassage
        s_passages = list(dict.fromkeys(r.get("samepassage", "") for r in rows if r.get("samepassage") is not None))
        # URLs
        urls = list(dict.fromkeys(r.get("url", "") for r in rows if r.get("url") is not None))
        # Question IDs
        q_ids = sorted(list(set(r.get("question_ID") for r in rows if r.get("question_ID") is not None)))

        intermediate_records.append({
            "context1": c1,
            "context2": c2,
            "titles": titles,
            "c_types": c_types,
            "s_passages": s_passages,
            "urls": urls,
            "question_IDs": q_ids,
            "rows": rows
        })

    # Sort records by article title, then context1
    def get_sort_key(rec):
        first_title = rec["titles"][0] if rec["titles"] else ""
        return (first_title, rec["context1"])

    intermediate_records.sort(key=get_sort_key)

    # Assign pair_id (integer) and check for disagreements inside pairs
    final_records = []
    conflicts_detected = []
    for idx, rec in enumerate(intermediate_records, start=1):
        pair_id = idx

        # Disagreement check
        title_val = rec["titles"][0] if len(rec["titles"]) == 1 else rec["titles"]
        if len(rec["titles"]) > 1:
            conflicts_detected.append((pair_id, "WikipediaArticleTitle", rec["titles"]))
            log(f"Warning: Disagreement in pair_id {pair_id} for WikipediaArticleTitle: {rec['titles']}")

        ctype_val = rec["c_types"][0] if len(rec["c_types"]) == 1 else rec["c_types"]
        if len(rec["c_types"]) > 1:
            conflicts_detected.append((pair_id, "contradictType", rec["c_types"]))
            log(f"Warning: Disagreement in pair_id {pair_id} for contradictType: {rec['c_types']}")

        sp_val = rec["s_passages"][0] if len(rec["s_passages"]) == 1 else rec["s_passages"]
        if len(rec["s_passages"]) > 1:
            conflicts_detected.append((pair_id, "samepassage", rec["s_passages"]))
            log(f"Warning: Disagreement in pair_id {pair_id} for samepassage: {rec['s_passages']}")

        url_val = rec["urls"][0] if len(rec["urls"]) == 1 else (rec["urls"] if len(rec["urls"]) > 1 else "")

        final_records.append({
            "pair_id": pair_id,
            "context1": rec["context1"],
            "context2": rec["context2"],
            "WikipediaArticleTitle": title_val,
            "url": url_val,
            "question_IDs": rec["question_IDs"],
            "contradictType": ctype_val,
            "samepassage": sp_val
        })

    # Write data/positives.jsonl
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved {len(final_records)} records to {OUTPUT_JSONL_PATH}")

    # Generate Report Sections
    log("\n" + "=" * 70)
    log("POSITIVES REPORT")
    log("=" * 70)

    # 1. Row counts before and after dedup
    log("\n1. ROW COUNTS (EXACT DEDUPLICATION)")
    log("-" * 40)
    log(f"Rows before deduplication: {raw_count}")
    log(f"Rows after deduplication:  {deduped_count}")
    log(f"Exact duplicate rows dropped: {raw_count - deduped_count}")

    # 2. Unique question_IDs
    log("\n2. UNIQUE QUESTION_IDS")
    log("-" * 40)
    all_q_ids = set()
    for r in final_records:
        all_q_ids.update(r["question_IDs"])
    log(f"Total unique question_IDs: {len(all_q_ids)}")

    # 3. Unique ordered and unordered pairs
    log("\n3. UNIQUE ORDERED AND UNORDERED PAIRS")
    log("-" * 40)
    log(f"Unique ordered (context1, context2) pairs:   {unique_ordered_pairs_count}")
    log(f"Unique unordered {{a, b}} pairs:              {unique_unordered_pairs_count}")
    if unique_ordered_pairs_count == unique_unordered_pairs_count:
        log("No reversed pairs (b, a) exist; ordered and unordered counts are identical.")
    else:
        log(f"{unique_ordered_pairs_count - unique_unordered_pairs_count} pairs appear in both orders.")

    # 4. Distribution of question_IDs per pair
    log("\n4. DISTRIBUTION OF QUESTION_IDS PER PAIR")
    log("-" * 40)
    q_counts = Counter(len(r["question_IDs"]) for r in final_records)
    for k in sorted(q_counts.keys()):
        count = q_counts[k]
        pct = (count / len(final_records)) * 100
        log(f"  - {k} question_ID(s) per pair: {count} pairs ({pct:.2f}%)")

    # 5. Pairs per article
    log("\n5. PAIRS PER ARTICLE")
    log("-" * 40)
    article_to_pairs = Counter()
    for r in final_records:
        title = r["WikipediaArticleTitle"] if isinstance(r["WikipediaArticleTitle"], str) else tuple(r["WikipediaArticleTitle"])
        article_to_pairs[title] += 1
    pair_counts_per_art = list(article_to_pairs.values())
    min_ppa = int(np.min(pair_counts_per_art))
    med_ppa = float(np.median(pair_counts_per_art))
    max_ppa = int(np.max(pair_counts_per_art))
    gte_2_ppa = sum(1 for c in pair_counts_per_art if c >= 2)
    log(f"Total unique articles: {len(article_to_pairs)}")
    log(f"Pairs per article - Min: {min_ppa}, Median: {med_ppa:.1f}, Max: {max_ppa}")
    log(f"Number of articles with at least 2 pairs: {gte_2_ppa}")

    # 6. contradictType and samepassage counts at pair level
    log("\n6. PAIR-LEVEL LABEL DISTRIBUTIONS")
    log("-" * 40)
    log("contradictType counts (pair level):")
    ct_counts = Counter(r["contradictType"] if isinstance(r["contradictType"], str) else str(r["contradictType"]) for r in final_records)
    for k, v in ct_counts.most_common():
        pct = (v / len(final_records)) * 100
        log(f"  - {k}: {v} ({pct:.2f}%)")

    log("\nsamepassage counts (pair level):")
    sp_counts = Counter(r["samepassage"] if isinstance(r["samepassage"], str) else str(r["samepassage"]) for r in final_records)
    for k, v in sp_counts.most_common():
        pct = (v / len(final_records)) * 100
        log(f"  - {k}: {v} ({pct:.2f}%)")

    # 7. Pair token counts with cross-encoder/nli-deberta-v3-base tokenizer
    log("\n7. PAIR TOKEN COUNTS (cross-encoder/nli-deberta-v3-base)")
    log("-" * 40)
    pair_token_lengths = [len(tokenizer(r["context1"], r["context2"], truncation=False)["input_ids"]) for r in final_records]
    tok_min = int(np.min(pair_token_lengths))
    tok_med = float(np.median(pair_token_lengths))
    tok_p95 = float(np.percentile(pair_token_lengths, 95))
    tok_max = int(np.max(pair_token_lengths))
    tok_over_512 = sum(1 for t in pair_token_lengths if t > 512)
    log(f"Pair token counts - Min: {tok_min}, Median: {tok_med:.1f}, 95th percentile: {tok_p95:.1f}, Max: {tok_max}")
    log(f"Number of pairs exceeding 512 tokens: {tok_over_512} (out of {len(final_records)})")

    # 8. Equality or substring relationship
    log("\n8. EQUALITY AND SUBSTRING RELATIONSHIPS")
    log("-" * 40)
    exact_equal_count = sum(1 for r in final_records if r["context1"] == r["context2"])
    strict_substring_count = sum(1 for r in final_records if (r["context1"] != r["context2"]) and (r["context1"] in r["context2"] or r["context2"] in r["context1"]))
    any_substring_or_equal = sum(1 for r in final_records if r["context1"] in r["context2"] or r["context2"] in r["context1"])
    log(f"Pairs where context1 == context2: {exact_equal_count}")
    log(f"Pairs where one is a strict substring of the other: {strict_substring_count}")
    log(f"Total pairs where context1 equals context2 or one is a substring of the other: {any_substring_or_equal}")

    # 9. Hub metadata and Cache path
    log("\n9. DATASET HUB METADATA AND LOCAL CACHE")
    log("-" * 40)
    log(f"Dataset ID:         {DATASET_ID}")
    log(f"Hub commit SHA:     {hub_sha}")
    log(f"Hub last-modified:  {last_modified}")
    log("Files actually loaded from cache:")
    if cache_files:
        for cf in cache_files:
            log(f"  - {cf.get('filename')}")
    else:
        log("  None / in-memory")

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nReport successfully saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
