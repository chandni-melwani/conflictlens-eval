import os
import sys
from collections import Counter
import numpy as np

# Ensure CPU only and clean HF logging
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPORT_PATH = os.path.join("reports", "dataset_inspection.txt")
DATASET_ID = "ibm/Wikipedia_contradict_benchmark"
TOKENIZER_ID = "cross-encoder/nli-deberta-v3-base"

lines = []

def log(text=""):
    print(text)
    lines.append(text)

def truncate(val, max_len=150):
    s = str(val) if val is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."

def main():
    os.makedirs("reports", exist_ok=True)

    try:
        from datasets import load_dataset
        log(f"Loading dataset '{DATASET_ID}'...")
        ds_dict = load_dataset(DATASET_ID)
    except Exception as e:
        print(f"Error loading dataset '{DATASET_ID}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
        log(f"Loading tokenizer '{TOKENIZER_ID}'...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        print(f"Error loading tokenizer '{TOKENIZER_ID}': {e}", file=sys.stderr)
        sys.exit(1)

    log("=" * 60)
    log("WIKICONTRADICT DATASET INSPECTION REPORT")
    log("=" * 60)

    # 1. Splits and count
    log("\n1. SPLITS AND ROW COUNT")
    log("-" * 30)
    for split_name, split_data in ds_dict.items():
        log(f"Split '{split_name}': {len(split_data)} rows")

    primary_split = list(ds_dict.keys())[0]
    split_data = ds_dict[primary_split]

    # 2. Fields
    log("\n2. FIELDS")
    log("-" * 30)
    features = split_data.features
    for col in split_data.column_names:
        feat_type = features[col].dtype if hasattr(features[col], "dtype") else type(features[col]).__name__
        ex_val = truncate(split_data[0][col], 150)
        log(f"- {col} (type: {feat_type}): {ex_val}")

    # 3. Example rows
    log("\n3. EXAMPLE ROWS (2 full rows, truncated to 300 chars)")
    log("-" * 30)
    for i in range(min(2, len(split_data))):
        log(f"\n--- Row {i + 1} ---")
        row = split_data[i]
        for col in split_data.column_names:
            log(f"  {col}: {truncate(row[col], 300)}")

    # 4. Label-like columns
    log("\n4. LABEL-LIKE COLUMNS")
    log("-" * 30)
    label_keywords = ["type", "label", "contradict", "conflict", "explicit", "implicit"]
    label_cols = [c for c in split_data.column_names if any(k in c.lower() for k in label_keywords)]
    if not label_cols:
        log("No label-like columns found.")
    for col in label_cols:
        log(f"Column '{col}':")
        counts = Counter()
        for row in split_data:
            val = row[col]
            if isinstance(val, (list, tuple)):
                counts.update(val)
            else:
                counts[val] += 1
        for val, count in counts.most_common():
            log(f"  - '{val}': {count}")

    # 5. Non-contradicting pairs
    log("\n5. NON-CONTRADICTING PAIRS")
    log("-" * 30)
    non_contra_indicators = {"no", "none", "false", "0", "consistent", "non-contradiction", "entailment", "neutral"}
    non_contra_found = False
    for col in label_cols:
        matches = 0
        matching_vals = set()
        for row in split_data:
            val = row[col]
            val_items = val if isinstance(val, (list, tuple)) else [val]
            for item in val_items:
                item_str = str(item).strip().lower()
                if item_str in non_contra_indicators or any(ind in item_str for ind in ["consistent", "non-contra"]):
                    matches += 1
                    matching_vals.add(str(item))
                    break
        if matches > 0:
            non_contra_found = True
            log(f"Column '{col}' suggests non-contradiction with values {matching_vals}: {matches} rows")
    if not non_contra_found:
        log("No explicit non-contradiction label found")

    # 6. Title or URL columns
    log("\n6. TITLE OR URL COLUMNS")
    log("-" * 30)
    meta_keywords = ["title", "url", "article", "link"]
    meta_cols = [c for c in split_data.column_names if any(k in c.lower() for k in meta_keywords)]
    if not meta_cols:
        log("No title or URL columns found.")
    for col in meta_cols:
        log(f"Column '{col}':")
        for idx in range(min(3, len(split_data))):
            ex = truncate(split_data[idx][col], 150)
            log(f"  Example {idx + 1}: {ex}")

    # 7. Token lengths
    log("\n7. TOKEN LENGTHS")
    log("-" * 30)
    text_cols = [c for c in split_data.column_names if len(split_data) > 0 and isinstance(split_data[0][c], str)]
    avg_lens = {c: float(np.mean([len(r[c] or "") for r in split_data])) for c in text_cols}
    sorted_cols = sorted(avg_lens.items(), key=lambda x: x[1], reverse=True)
    log("Text columns ranked by average character length:")
    for c, l in sorted_cols:
        log(f"  - {c}: {l:.1f} chars avg")

    def analyze_pair(c1, c2, note=""):
        log(f"\nEvaluating passage pair: '{c1}' and '{c2}' {note}")
        t1 = [len(tokenizer(r[c1] or "", truncation=False)["input_ids"]) for r in split_data]
        t2 = [len(tokenizer(r[c2] or "", truncation=False)["input_ids"]) for r in split_data]
        t_pair = [len(tokenizer(r[c1] or "", r[c2] or "", truncation=False)["input_ids"]) for r in split_data]
        for name, arr in [(f"Passage 1 ('{c1}')", t1), (f"Passage 2 ('{c2}')", t2), ("Pair encoded together", t_pair)]:
            log(f"  {name}: min={int(np.min(arr))}, median={float(np.median(arr)):.1f}, "
                f"95th={float(np.percentile(arr, 95)):.1f}, max={int(np.max(arr))}")
        exceed_512 = sum(1 for t in t_pair if t > 512)
        log(f"  Rows where pair exceeds 512 tokens: {exceed_512} (out of {len(split_data)})")

    p1_col, p2_col = sorted_cols[0][0], sorted_cols[1][0]
    log(f"\nTwo longest text columns chosen: '{p1_col}' and '{p2_col}'")
    analyze_pair(p1_col, p2_col, "(strictly two longest text columns)")

    # Also report individual passage columns if merged_context was one of the two longest
    if "merged_context" in (p1_col, p2_col) and "context1" in split_data.column_names and "context2" in split_data.column_names:
        analyze_pair("context1", "context2", "(unmerged distinct passage columns)")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nReport successfully saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
