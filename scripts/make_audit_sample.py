"""
scripts/make_audit_sample.py

Build the extraction audit sample (Spec 15).
Draws 50 test-split passages and their cached extractions for a manual audit.
Read-only on cache. No Ollama, no model runs, no prompt changes.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.extract import get_cache_path


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_audit_sample(
    gold_path: str = os.path.join("data", "gold.jsonl"),
    cache_dir: str = os.path.join("data", "cache", "extractions"),
    sample_out: str = os.path.join("data", "extraction_audit_sample.jsonl"),
    key_out: str = os.path.join("data", "extraction_audit_key.jsonl"),
    plan_out: str = os.path.join("reports", "extraction_audit_sample_plan.txt"),
    seed: int = 42,
    model: str = "qwen2.5:3b",
    prompt_version: str = "extract_v2",
) -> dict:
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]

    # Filter test split only
    test_items = [x for x in gold if x.get("split") == "test"]
    pos_items = [x for x in test_items if x.get("label") == 1]
    neg_items = [x for x in test_items if x.get("label") == 0]

    if len(pos_items) < 25:
        raise ValueError(f"Need at least 25 positive test items, found {len(pos_items)}")
    if len(neg_items) < 25:
        raise ValueError(f"Need at least 25 negative test items, found {len(neg_items)}")

    # Sort deterministically before sampling
    sorted_pos = sorted(pos_items, key=lambda x: str(x["id"]))
    sorted_neg = sorted(neg_items, key=lambda x: str(x["id"]))

    rng = random.Random(seed)

    # Draw 25 passages from positive items and 25 from negative items,
    # at random, at most one passage per item.
    sampled_pos_items = rng.sample(sorted_pos, 25)
    pos_draw = []
    for it in sampled_pos_items:
        p_idx = rng.choice([1, 2])
        pos_draw.append((it, p_idx))

    sampled_neg_items = rng.sample(sorted_neg, 25)
    neg_draw = []
    for it in sampled_neg_items:
        p_idx = rng.choice([1, 2])
        neg_draw.append((it, p_idx))

    # Look up extractions in cache using existing cache key function
    # If any sampled passage has no cache entry, stop and report which ones.
    all_draw = pos_draw + neg_draw
    missing_cache = []
    extracted_data = {}

    for it, p_idx in all_draw:
        passage_text = it[f"passage{p_idx}"]
        c_path = get_cache_path(passage_text, cache_dir=cache_dir, model=model, prompt_version=prompt_version)
        if not os.path.exists(c_path):
            missing_cache.append((it["id"], p_idx, c_path))
        else:
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            extracted_data[(it["id"], p_idx)] = c_data.get("facts", [])

    if missing_cache:
        err_lines = [
            f"Missing extraction cache entry for item '{item_id}', passage {p_idx} (path: {cp})"
            for item_id, p_idx, cp in missing_cache
        ]
        raise RuntimeError("Missing extraction cache entries for sampled passages:\n" + "\n".join(err_lines))

    # Shuffle the 50 and assign audit ids a01 to a50
    combined = list(all_draw)
    rng.shuffle(combined)

    sample_rows = []
    key_rows = []

    for idx, (it, p_idx) in enumerate(combined, start=1):
        audit_id = f"a{idx:02d}"
        passage_text = it[f"passage{p_idx}"]
        facts = extracted_data[(it["id"], p_idx)]

        sample_row = {
            "audit_id": audit_id,
            "passage_text": passage_text,
            "facts": facts,
        }
        sample_rows.append(sample_row)

        key_row = {
            "audit_id": audit_id,
            "item_id": it["id"],
            "passage_index": p_idx,
            "gold_label": it["label"],
        }
        key_rows.append(key_row)

    # Write data/extraction_audit_sample.jsonl
    os.makedirs(os.path.dirname(sample_out) or ".", exist_ok=True)
    with open(sample_out, "w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write data/extraction_audit_key.jsonl
    os.makedirs(os.path.dirname(key_out) or ".", exist_ok=True)
    with open(key_out, "w", encoding="utf-8") as f:
        for r in key_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Metrics for the plan
    fact_counts = [len(r["facts"]) for r in sample_rows]
    zero_facts_count = sum(1 for c in fact_counts if c == 0)
    median_facts = statistics.median(fact_counts)
    max_facts = max(fact_counts)

    sample_sha256 = compute_sha256(sample_out)
    key_sha256 = compute_sha256(key_out)

    sample_out_display = sample_out.replace("\\", "/")
    key_out_display = key_out.replace("\\", "/")

    os.makedirs(os.path.dirname(plan_out) or ".", exist_ok=True)
    plan_lines = [
        "Extraction Audit Sampling Plan (Spec 15)",
        "========================================",
        "",
        "Sampling Parameters:",
        f"  Random seed: {seed}",
        f"  Gold test pool size: {len(test_items)} items ({len(pos_items)} positive, {len(neg_items)} negative)",
        f"  Passages drawn: 50 (25 positive items, 25 negative items, at most one passage per item)",
        "",
        "Fact Counts Across Sampled Passages:",
        f"  Passages with 0 facts: {zero_facts_count}",
        f"  Median facts per passage: {median_facts}",
        f"  Max facts per passage: {max_facts}",
        "",
        "File SHA-256 Checksums:",
        f"  {sample_out_display}: {sample_sha256}",
        f"  {key_out_display}: {key_sha256}",
    ]
    plan_content = "\n".join(plan_lines) + "\n"
    with open(plan_out, "w", encoding="utf-8") as f:
        f.write(plan_content)

    return {
        "sample_rows": len(sample_rows),
        "key_rows": len(key_rows),
        "zero_facts_count": zero_facts_count,
        "median_facts": median_facts,
        "max_facts": max_facts,
        "sample_sha256": sample_sha256,
        "key_sha256": key_sha256,
        "plan_content": plan_content,
    }


def main():
    parser = argparse.ArgumentParser(description="Build extraction audit sample (Spec 15)")
    parser.add_argument("--gold", default=os.path.join("data", "gold.jsonl"), help="Path to gold.jsonl")
    parser.add_argument(
        "--cache_dir",
        default=os.path.join("data", "cache", "extractions"),
        help="Extraction cache dir",
    )
    parser.add_argument(
        "--sample_out",
        default=os.path.join("data", "extraction_audit_sample.jsonl"),
        help="Output sample jsonl",
    )
    parser.add_argument(
        "--key_out",
        default=os.path.join("data", "extraction_audit_key.jsonl"),
        help="Output key jsonl",
    )
    parser.add_argument(
        "--plan_out",
        default=os.path.join("reports", "extraction_audit_sample_plan.txt"),
        help="Output plan report",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", default="qwen2.5:3b", help="Model name for cache key")
    parser.add_argument("--prompt_version", default="extract_v2", help="Prompt version for cache key")
    args = parser.parse_args()

    try:
        res = build_audit_sample(
            gold_path=args.gold,
            cache_dir=args.cache_dir,
            sample_out=args.sample_out,
            key_out=args.key_out,
            plan_out=args.plan_out,
            seed=args.seed,
            model=args.model,
            prompt_version=args.prompt_version,
        )
        print(res["plan_content"])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
