import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.extract import MODEL, extract_facts
from scripts.metrics_extract import compute_extraction_metrics

DATA_PATH = os.path.join("data", "positives_split.jsonl")
REPORT_PATH = os.path.join("reports", "extraction_compare_dev.txt")


def load_dev_pairs():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    dev_records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Only dev positives, two_passage stratum. Never touch test or negatives.
            if rec.get("split") == "dev" and rec.get("stratum") == "two_passage":
                dev_records.append(rec)

    dev_records.sort(key=lambda x: x["pair_id"])
    # Positions are 1-based; reserve positions 21 to 26 remain untouched.
    return dev_records[:20]


def run_extractions_for_slice(pairs, prompt_version: str):
    results = []
    for rec in pairs:
        pid = rec["pair_id"]
        for p_label, passage in [("p1", rec["context1"]), ("p2", rec["context2"])]:
            res = extract_facts(passage, prompt_version=prompt_version)
            results.append({
                "pair_id": pid,
                "passage_label": p_label,
                "passage": passage,
                "res": res,
                "parse_ok": res["parse_ok"],
                "facts": res["facts"],
                "latency": res["latency"],
                "cached": res.get("cached", False),
            })
    return results


def run_comparison(prompt_version=None, pairs_arg=None):
    os.makedirs("reports", exist_ok=True)
    all_dev = load_dev_pairs()

    if prompt_version is not None and pairs_arg is not None:
        start_s, end_s = pairs_arg.split(":")
        start_i, end_i = int(start_s), int(end_s)
        selected_pairs = all_dev[start_i - 1 : end_i]
        runs = [(prompt_version, pairs_arg, selected_pairs)]
    else:
        # Full comparison as specified in SPEC 05:
        # v1 on 1:10 (cached), v1 on 11:20 (new calls), v2 on 1:10, v2 on 11:20
        pairs_1_10 = all_dev[0:10]
        pairs_11_20 = all_dev[10:20]
        runs = [
            ("extract_v1", "1:10", pairs_1_10),
            ("extract_v1", "11:20", pairs_11_20),
            ("extract_v2", "1:10", pairs_1_10),
            ("extract_v2", "11:20", pairs_11_20),
        ]

    report_lines = []

    def out(text=""):
        try:
            print(text)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            print(text.encode(encoding, errors="replace").decode(encoding))
        report_lines.append(text)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out("=" * 95)
    out("PROMPT EXTRACTION COMPARISON: DEV POSITIVES")
    out("=" * 95)
    out(f"Run Date: {run_date}")
    out(f"Model:    {MODEL}")
    out("-" * 95)

    # Header for comparison table
    header = (
        f"{'Prompt Version':<14} {'Subset':<8} {'Passages':<9} "
        f"{'Parse Fail':<12} {'Zero Fact':<11} {'Verbatim Val':<14} "
        f"{'Numeric Ent':<13} {'Med Facts':<11} {'Med Latency':<12}"
    )
    out(header)
    out("-" * 95)

    stored_v2_11_20_results = []

    for ver, subset_name, pair_slice in runs:
        results = run_extractions_for_slice(pair_slice, prompt_version=ver)
        if ver == "extract_v2" and subset_name == "11:20":
            stored_v2_11_20_results = results

        metrics = compute_extraction_metrics(results)
        num_passages = len(results)

        pf_str = f"{metrics['parse_fail_rate']:.3f}"
        zf_str = f"{metrics['zero_fact_rate']:.3f}"
        vv_str = f"{metrics['verbatim_value_rate']:.3f}"
        ne_str = f"{metrics['numeric_entity_rate']:.3f}"
        mf_str = f"{metrics['median_facts']:.1f}"

        med_lat = metrics['median_latency']
        if med_lat is not None:
            # Check if all were cached
            all_cached = all(r["cached"] for r in results)
            lat_str = f"{med_lat:.2f}s" + (" [c]" if all_cached else "")
        else:
            lat_str = "N/A"

        row = (
            f"{ver:<14} {subset_name:<8} {num_passages:<9} "
            f"{pf_str:<12} {zf_str:<11} {vv_str:<14} "
            f"{ne_str:<13} {mf_str:<11} {lat_str:<12}"
        )
        out(row)

    out("-" * 95)
    out("[c] indicates all calls were loaded from cache; latency reflects stored execution time.")

    # Five full extractions from the v2 run on positions 11 to 20
    if stored_v2_11_20_results:
        out("\n" + "=" * 95)
        out("FIVE FULL EXTRACTIONS: extract_v2 ON POSITIONS 11 TO 20")
        out("=" * 95)

        for idx, item in enumerate(stored_v2_11_20_results[:5], start=1):
            pid = item["pair_id"]
            p_lbl = item["passage_label"]
            passage = item["passage"]
            res = item["res"]

            trunc_passage = passage[:300] + ("..." if len(passage) > 300 else "")
            cached_flag = " [CACHE]" if res.get("cached", False) else ""

            out(f"\n--- Extraction {idx} (Pair ID: {pid}, Passage: {p_lbl}) ---")
            out("Source Passage (truncated to 300 characters):")
            out(trunc_passage)
            out(f"\nParse OK: {res['parse_ok']}, Fact Count: {len(res['facts'])}, Latency: {res['latency']:.2f}s{cached_flag}")
            out("Facts:")
            if res["facts"]:
                out(json.dumps(res["facts"], indent=2, ensure_ascii=False))
            else:
                out("  []")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport written to {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Compare extraction prompts on dev positives.")
    parser.add_argument("--prompt_version", type=str, default=None, help="Prompt version (extract_v1, extract_v2)")
    parser.add_argument("--pairs", type=str, default=None, help="1-based START:END slice in sorted dev list, e.g. 1:10")
    args = parser.parse_args()

    run_comparison(prompt_version=args.prompt_version, pairs_arg=args.pairs)


if __name__ == "__main__":
    main()
