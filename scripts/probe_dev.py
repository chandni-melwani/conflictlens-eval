import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.extract import MODEL, PROMPT_VERSION, extract_facts

DATA_PATH = os.path.join("data", "positives_split.jsonl")
REPORT_PATH = os.path.join("reports", "extraction_probe_dev.txt")


def run_probe(prompt_version: str = PROMPT_VERSION, pairs_slice: str = "1:10"):
    os.makedirs("reports", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    dev_records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") == "dev" and rec.get("stratum") == "two_passage":
                dev_records.append(rec)

    dev_records.sort(key=lambda x: x["pair_id"])
    start_s, end_s = pairs_slice.split(":")
    start_pos, end_pos = int(start_s), int(end_s)
    pairs = dev_records[start_pos - 1 : end_pos]

    report_lines = []

    def out(text=""):
        try:
            print(text)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            print(text.encode(encoding, errors="replace").decode(encoding))
        report_lines.append(text)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out("=" * 70)
    out(f"EXTRACTION PROBE: DEV POSITIVES ({len(pairs)} PAIRS, {pairs_slice})")
    out("=" * 70)
    out(f"Run Date:       {run_date}")
    out(f"MODEL:          {MODEL}")
    out(f"PROMPT_VERSION: {prompt_version}")
    out("-" * 70)
    out(f"{'Pair ID':<10} {'Passage':<10} {'Facts':<8} {'Parse OK':<10} {'Latency':<16}")
    out("-" * 70)

    results = []
    examples = []

    for rec in pairs:
        pid = rec["pair_id"]
        for p_label, passage in [("p1", rec["context1"]), ("p2", rec["context2"])]:
            res = extract_facts(passage, prompt_version=prompt_version)
            is_cached = res.get("cached", False)
            lat_str = f"{res['latency']:.2f}s" + (" [CACHE]" if is_cached else "")
            out(f"{pid:<10} {p_label:<10} {len(res['facts']):<8} {str(res['parse_ok']):<10} {lat_str:<16}")
            results.append({
                "pair_id": pid,
                "passage_label": p_label,
                "passage": passage,
                "res": res,
            })
            if len(examples) < 3:
                examples.append((pid, p_label, passage, res))

    parse_failures = sum(1 for r in results if not r["res"]["parse_ok"])
    non_cached_latencies = [r["res"]["latency"] for r in results if not r["res"].get("cached", False)]

    if non_cached_latencies:
        median_lat = statistics.median(non_cached_latencies)
    else:
        all_latencies = [r["res"]["latency"] for r in results]
        median_lat = statistics.median(all_latencies) if all_latencies else 0.0

    est_total_time = median_lat * 433

    out("-" * 70)
    out("SUMMARY METRICS")
    out(f"Total Passages Processed: {len(results)}")
    out(f"Number of Parse Failures: {parse_failures}")
    if non_cached_latencies:
        out(f"Non-cached Calls Count:   {len(non_cached_latencies)}")
        out(f"Median Latency (non-cached): {median_lat:.2f}s")
    else:
        out(f"All calls were cached. Cached Median Latency: {median_lat:.2f}s")
    out(f"Estimated Total Time (433 passages): {est_total_time:.2f}s ({est_total_time / 60:.2f} min)")
    out("-" * 70)

    out("\nTHREE EXAMPLE EXTRACTIONS IN FULL")
    out("=" * 70)
    for idx, (pid, p_label, passage, res) in enumerate(examples, start=1):
        trunc_passage = passage[:300] + ("..." if len(passage) > 300 else "")
        out(f"\n--- Example {idx} (Pair ID: {pid}, Passage: {p_label}) ---")
        out(f"Source Passage (first 300 chars):\n{trunc_passage}\n")
        out(f"Parse OK: {res['parse_ok']}, Number of Facts: {len(res['facts'])}, Latency: {res['latency']:.2f}s")
        out("Extracted Facts:")
        if res["facts"]:
            out(json.dumps(res["facts"], indent=2, ensure_ascii=False))
        else:
            out("  []")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe dev set extractions.")
    parser.add_argument("--prompt_version", type=str, default=PROMPT_VERSION, help="Prompt version (extract_v1, extract_v2)")
    parser.add_argument("--pairs", type=str, default="1:10", help="1-based START:END slice in sorted dev list, e.g. 1:10")
    args = parser.parse_args()
    run_probe(prompt_version=args.prompt_version, pairs_slice=args.pairs)

