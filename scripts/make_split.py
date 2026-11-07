import os
import sys
import json
import random
from collections import defaultdict, Counter
import numpy as np

# Force CPU only & disable symlinks warning
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_JSONL = os.path.join("data", "positives.jsonl")
OUTPUT_SPLIT_JSON = os.path.join("splits", "split_v1.json")
OUTPUT_SPLIT_JSONL = os.path.join("data", "positives_split.jsonl")
REPORT_PATH = os.path.join("reports", "split_report.txt")
TOKENIZER_ID = "cross-encoder/nli-deberta-v3-base"
RANDOM_SEED = 42

lines = []

def log(text=""):
    print(text)
    lines.append(text)

def main():
    os.makedirs("splits", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    log("=" * 70)
    log("SPLIT CREATION: WIKICONTRADICT BENCHMARK")
    log("=" * 70)

    if not os.path.exists(INPUT_JSONL):
        print(f"Error: {INPUT_JSONL} not found.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    log(f"Loaded {len(records)} pairs from {INPUT_JSONL}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Report list-valued or conflicting contradictType, samepassage or title
    conflicting_pairs = []
    for r in records:
        issues = {}
        for field in ["contradictType", "samepassage", "WikipediaArticleTitle"]:
            val = r.get(field)
            if isinstance(val, (list, tuple)):
                issues[field] = val
        if issues:
            conflicting_pairs.append((r.get("pair_id"), issues))

    # 2. Add identical = (context1 == context2) and find disagreements with samepassage == "Same"
    disagreements = []
    for r in records:
        r["identical"] = (r["context1"] == r["context2"])
        sp_is_same = (str(r.get("samepassage", "")).strip().lower() == "same")
        if r["identical"] != sp_is_same:
            disagreements.append(r)

    # 3. Stratum assignment
    # Stratum: same_passage if identical, else two_passage
    for r in records:
        r["stratum"] = "same_passage" if r["identical"] else "two_passage"

    # 4. Split two_passage pairs by WikipediaArticleTitle, seed 42, ~25% pairs to dev
    # All same_passage pairs get split same_passage_analysis
    two_passage_pairs = [r for r in records if r["stratum"] == "two_passage"]
    same_passage_pairs = [r for r in records if r["stratum"] == "same_passage"]

    art_to_two_passage = defaultdict(list)
    for r in two_passage_pairs:
        art_to_two_passage[r["WikipediaArticleTitle"]].append(r)

    two_passage_articles = sorted(list(art_to_two_passage.keys()))
    rng = random.Random(RANDOM_SEED)
    shuffled_articles = list(two_passage_articles)
    rng.shuffle(shuffled_articles)

    target_dev_pairs = round(len(two_passage_pairs) * 0.25)
    dev_articles = set()
    current_dev_pairs = 0
    for art in shuffled_articles:
        art_pairs = len(art_to_two_passage[art])
        if current_dev_pairs + art_pairs <= target_dev_pairs or current_dev_pairs == 0:
            dev_articles.add(art)
            current_dev_pairs += art_pairs
            if current_dev_pairs >= target_dev_pairs:
                break
        else:
            # Check if adding this article gets closer to target than stopping
            if abs(current_dev_pairs + art_pairs - target_dev_pairs) < abs(current_dev_pairs - target_dev_pairs):
                dev_articles.add(art)
                current_dev_pairs += art_pairs
            break

    # Assign split labels
    for r in records:
        if r["stratum"] == "same_passage":
            r["split"] = "same_passage_analysis"
        elif r["WikipediaArticleTitle"] in dev_articles:
            r["split"] = "dev"
        else:
            r["split"] = "test"

    # 5. Write splits/split_v1.json (article titles, pair_ids, split labels, seed; NO passage text)
    splits_dict = {
        "seed": RANDOM_SEED,
        "splits": {
            "dev": {
                "article_titles": sorted(list(dev_articles)),
                "pair_ids": sorted([r["pair_id"] for r in records if r["split"] == "dev"])
            },
            "test": {
                "article_titles": sorted(list(set(r["WikipediaArticleTitle"] for r in records if r["split"] == "test"))),
                "pair_ids": sorted([r["pair_id"] for r in records if r["split"] == "test"])
            },
            "same_passage_analysis": {
                "article_titles": sorted(list(set(r["WikipediaArticleTitle"] for r in records if r["split"] == "same_passage_analysis"))),
                "pair_ids": sorted([r["pair_id"] for r in records if r["split"] == "same_passage_analysis"])
            }
        },
        "pairs": [
            {
                "pair_id": r["pair_id"],
                "WikipediaArticleTitle": r["WikipediaArticleTitle"],
                "split": r["split"],
                "stratum": r["stratum"],
                "identical": r["identical"]
            }
            for r in records
        ]
    }

    with open(OUTPUT_SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2, ensure_ascii=False)
    log(f"Saved split metadata to {OUTPUT_SPLIT_JSON}")

    # Write data/positives_split.jsonl
    with open(OUTPUT_SPLIT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved full split records to {OUTPUT_SPLIT_JSONL}")

    # Generate Report
    log("\n" + "=" * 70)
    log("SPLIT REPORT")
    log("=" * 70)

    # Item 1: Conflicting or list-valued columns
    log("\n1. CONFLICTING OR LIST-VALUED ATTRIBUTES")
    log("-" * 45)
    log(f"Number of pairs with list-valued or conflicting contradictType, samepassage, or title: {len(conflicting_pairs)}")
    if conflicting_pairs:
        for pid, issues in conflicting_pairs:
            log(f"  - pair_id {pid}: {issues}")
    else:
        log("  None found (all pairs have unique scalar values).")

    # Item 2: identical vs samepassage disagreements
    log("\n2. DISAGREEMENTS: IDENTICAL VS SAMEPASSAGE == 'SAME'")
    log("-" * 45)
    log(f"Number of pairs where identical disagrees with (samepassage == 'Same'): {len(disagreements)}")
    if disagreements:
        log("\nFull details for disagreeing pair(s):")
        for d in disagreements:
            log(json.dumps(d, indent=2, ensure_ascii=False))
    else:
        log("  None found.")

    # Item 3: Pairs per split and stratum
    log("\n3. PAIRS PER SPLIT AND STRATUM")
    log("-" * 45)
    split_names = ["dev", "test", "same_passage_analysis"]
    strata = ["two_passage", "same_passage"]

    header = f"{'Split':<25} | {'two_passage':>12} | {'same_passage':>12} | {'Total':>8}"
    sep = "-" * len(header)
    log(header)
    log(sep)
    for s in split_names:
        c_two = sum(1 for r in records if r["split"] == s and r["stratum"] == "two_passage")
        c_same = sum(1 for r in records if r["split"] == s and r["stratum"] == "same_passage")
        total = c_two + c_same
        log(f"{s:<25} | {c_two:>12} | {c_same:>12} | {total:>8}")
    log(sep)
    tot_two = sum(1 for r in records if r["stratum"] == "two_passage")
    tot_same = sum(1 for r in records if r["stratum"] == "same_passage")
    log(f"{'Total':<25} | {tot_two:>12} | {tot_same:>12} | {len(records):>8}")

    # Item 4: Explicit / Implicit counts per split
    log("\n4. EXPLICIT / IMPLICIT COUNTS PER SPLIT")
    log("-" * 45)
    header_ct = f"{'Split':<25} | {'Explicit':>10} | {'Implicit':>10} | {'Total':>8} | {'% Explicit':>10}"
    sep_ct = "-" * len(header_ct)
    log(header_ct)
    log(sep_ct)
    for s in split_names:
        sub = [r for r in records if r["split"] == s]
        ct_counts = Counter(r["contradictType"] for r in sub)
        exp = ct_counts.get("Explicit", 0)
        imp = ct_counts.get("Implicit (reasoning required)", 0)
        tot = len(sub)
        pct = (exp / tot * 100) if tot > 0 else 0.0
        log(f"{s:<25} | {exp:>10} | {imp:>10} | {tot:>8} | {pct:>9.1f}%")
    log(sep_ct)
    all_exp = sum(1 for r in records if r["contradictType"] == "Explicit")
    all_imp = sum(1 for r in records if r["contradictType"] == "Implicit (reasoning required)")
    all_pct = (all_exp / len(records) * 100) if len(records) > 0 else 0.0
    log(f"{'Total':<25} | {all_exp:>10} | {all_imp:>10} | {len(records):>8} | {all_pct:>9.1f}%")

    # Item 5: Median pair token length per split
    log("\n5. MEDIAN PAIR TOKEN LENGTH PER SPLIT (cross-encoder/nli-deberta-v3-base)")
    log("-" * 45)
    for s in split_names:
        sub = [r for r in records if r["split"] == s]
        token_lengths = [len(tokenizer(r["context1"], r["context2"], truncation=False)["input_ids"]) for r in sub]
        med = float(np.median(token_lengths)) if token_lengths else 0.0
        min_len = int(np.min(token_lengths)) if token_lengths else 0
        max_len = int(np.max(token_lengths)) if token_lengths else 0
        p95 = float(np.percentile(token_lengths, 95)) if token_lengths else 0.0
        log(f"Split '{s}' ({len(sub)} pairs):")
        log(f"  Median token length: {med:.1f} (min: {min_len}, 95th: {p95:.1f}, max: {max_len})")

    # Item 6: Number of articles per split
    log("\n6. NUMBER OF ARTICLES PER SPLIT")
    log("-" * 45)
    for s in split_names:
        sub = [r for r in records if r["split"] == s]
        arts = set(r["WikipediaArticleTitle"] for r in sub)
        log(f"  - Split '{s}': {len(arts)} articles ({len(sub)} pairs)")
    all_arts = set(r["WikipediaArticleTitle"] for r in records)
    log(f"  - Total unique articles across all splits: {len(all_arts)} articles ({len(records)} pairs)")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nReport successfully saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()

