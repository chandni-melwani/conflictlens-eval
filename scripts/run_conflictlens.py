import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.cl_detect import check_pair, get_candidate_pairs
from scripts.extract import extract_facts
from scripts.metrics import confusion, mcc
from scripts.nli import contradiction_score

TYPE_MAP = {"date_time": "temporal", "number": "numerical", "other": "semantic"}
DEFAULT_GOLD = os.path.join("data", "gold.jsonl")
DEFAULT_GOLD_IDENTICAL = os.path.join("data", "gold_identical.jsonl")


def get_out_dir(variant: str, blocking: str, split: str, base_dir: str = "results") -> str:
    return os.path.join(base_dir, f"conflictlens_{variant}_{blocking}_{split}")


def select_items(split: str, gold: str = DEFAULT_GOLD) -> tuple[str, list[dict]]:
    gold_path = gold
    if split == "identical":
        if gold is None or os.path.normpath(gold) == os.path.normpath(DEFAULT_GOLD):
            gold_path = DEFAULT_GOLD_IDENTICAL
        target_split = "same_passage_analysis"
    else:
        target_split = split

    with open(gold_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    items = [it for it in items if it.get("split") == target_split]
    return gold_path, items


def print_stats(details: list[dict], name: str):
    n = len(details)
    if n == 0:
        return
    n_failed = sum(1 for d in details for ok in (d["p1_ok"], d["p2_ok"]) if not ok)
    fail_rate = n_failed / (2 * n)
    no_pairs = sum(1 for d in details if d["no_pairs"]) / n
    num_share = sum(1 for d in details if "numerical" in d["flagged_detectors"]) / n
    temp_share = sum(1 for d in details if "temporal" in d["flagged_detectors"]) / n
    sem_share = sum(1 for d in details if "semantic" in d["flagged_detectors"]) / n
    flagged_tps = [d for d in details if d["label"] == 1 and d["pred"] == 1 and d["flagged_detectors"]]
    match_cnt = sum(1 for d in flagged_tps if TYPE_MAP.get(d.get("type_group")) in d["flagged_detectors"])
    match_share = (match_cnt / len(flagged_tps)) if flagged_tps else 0.0

    print(f"\n--- Statistics for {name} ---")
    print(f"Parse-failure rate: {fail_rate:.4f} ({n_failed}/{2 * n})")
    print(f"Share of items with no candidate pairs: {no_pairs:.4f} ({sum(1 for d in details if d['no_pairs'])}/{n})")
    print(f"Share of items with >=1 flagged pair (Numerical): {num_share:.4f}")
    print(f"Share of items with >=1 flagged pair (Temporal):  {temp_share:.4f}")
    print(f"Share of items with >=1 flagged pair (Semantic):  {sem_share:.4f}")
    print(f"Flagged TPs matching gold type_group: {match_share:.4f} ({match_cnt}/{len(flagged_tps)})\n")


def main():
    p = argparse.ArgumentParser(description="Run ConflictLens pipeline")
    p.add_argument("--split", required=True, choices=["dev", "test", "identical"])
    p.add_argument("--variant", required=True, choices=["structured", "hybrid"])
    p.add_argument("--blocking", required=True, choices=["exact", "fuzzy", "none"])
    p.add_argument("--gold", default=DEFAULT_GOLD)
    p.add_argument("--no_numeric", action="store_true")
    p.add_argument("--no_temporal", action="store_true")
    p.add_argument("--no_semantic", action="store_true")
    args = p.parse_args()

    s_thresh_path = os.path.join("results", "conflictlens_structured_dev", "threshold.json")
    n_thresh_path = os.path.join("results", "nli_only_dev", "threshold.json")
    out_dir = get_out_dir(args.variant, args.blocking, args.split)

    if args.blocking != "exact":
        preds_file = os.path.join(out_dir, "preds.jsonl")
        if os.path.exists(preds_file):
            sys.stderr.write(f"Error: Output file already exists: {preds_file}\n")
            sys.exit(1)

    if args.blocking != "exact" or args.split in ("test", "identical") or args.variant == "hybrid":
        if not os.path.exists(s_thresh_path):
            sys.stderr.write(f"Error: Missing threshold file {s_thresh_path}\n")
            sys.exit(1)
        if args.variant == "hybrid" and not os.path.exists(n_thresh_path):
            sys.stderr.write(f"Error: Missing threshold file {n_thresh_path}\n")
            sys.exit(1)

    gold_path, items = select_items(args.split, args.gold)
    args.gold = gold_path

    if not items:
        sys.stderr.write(f"Error: Zero items selected for split '{args.split}' from {gold_path}\n")
        sys.exit(1)

    extracted = []
    for it in items:
        r1 = extract_facts(it["passage1"], prompt_version="extract_v2")
        r2 = extract_facts(it["passage2"], prompt_version="extract_v2")
        pairs, n_blocks = get_candidate_pairs(r1.get("facts", []), r2.get("facts", []), args.blocking)
        evaluated_pairs, t_nli = [], 0.0
        for f1, f2 in pairs:
            t0 = time.perf_counter()
            pair_res = check_pair(
                f1, f2, tau_sem=0.5, enable_numeric=not args.no_numeric,
                enable_temporal=not args.no_temporal, enable_semantic=not args.no_semantic
            )
            t_nli += time.perf_counter() - t0
            evaluated_pairs.append({**pair_res, "f1": f1, "f2": f2})
        extracted.append({
            "item": it, "r1": r1, "r2": r2, "pairs": evaluated_pairs, "n_blocks": n_blocks, "nli_time": t_nli
        })

    tau_sem = 0.5
    if args.split == "dev" and args.variant == "structured" and args.blocking == "exact":
        if os.path.exists(s_thresh_path):
            print(f"Threshold file {s_thresh_path} already exists; preserving existing threshold without writing.")
            with open(s_thresh_path, "r", encoding="utf-8") as f:
                tau_sem = json.load(f)["threshold"]
        else:
            thresholds = [round(i * 0.05, 2) for i in range(1, 20)]
            labels = [ex["item"]["label"] for ex in extracted]
            best_t, best_mcc, curve = thresholds[0], -2.0, []
            for t in thresholds:
                preds = []
                for ex in extracted:
                    fl = any(p["score"] == 1.0 if p["detector"] in ("numerical", "temporal") else p["score"] >= t for p in ex["pairs"])
                    preds.append(1 if fl else 0)
                val = mcc(tp=confusion(labels, preds)[0], fp=confusion(labels, preds)[1], fn=confusion(labels, preds)[2], tn=confusion(labels, preds)[3])
                curve.append({"threshold": t, "mcc": val})
                if val > best_mcc:
                    best_mcc, best_t = val, t
            tau_sem = best_t
            os.makedirs(os.path.dirname(s_thresh_path), exist_ok=True)
            with open(s_thresh_path, "w", encoding="utf-8") as f:
                json.dump({"threshold": best_t, "mcc": best_mcc, "curve": curve}, f, indent=2)
    else:
        with open(s_thresh_path, "r", encoding="utf-8") as f:
            tau_sem = json.load(f)["threshold"]

    tau_nli = None
    if args.variant == "hybrid":
        with open(n_thresh_path, "r", encoding="utf-8") as f:
            tau_nli = json.load(f)["threshold"]

    preds_rows, details_rows = [], []
    for ex in extracted:
        it, r1, r2, pairs = ex["item"], ex["r1"], ex["r2"], ex["pairs"]
        no_pairs = (len(pairs) == 0)
        flagged = [p for p in pairs if (p["score"] == 1.0 if p["detector"] in ("numerical", "temporal") else p["score"] >= tau_sem)]
        max_score = max((p["score"] for p in pairs), default=0.0)
        fallback_used = False
        if args.variant == "hybrid" and no_pairs:
            fallback_used = True
            t0 = time.perf_counter()
            max_score = contradiction_score(it["passage1"], it["passage2"])
            ex["nli_time"] += time.perf_counter() - t0
            pred = 1 if max_score >= tau_nli else 0
        else:
            pred = 1 if flagged else 0

        flagged_detectors = [p["detector"] for p in flagged]
        preds_rows.append({"id": it["id"], "pred": pred, "score": max_score})
        details_rows.append({
            "id": it["id"], "label": it.get("label"), "type_group": it.get("type_group"), "pred": pred,
            "score": max_score, "no_pairs": no_pairs, "fallback_used": fallback_used,
            "flagged_detectors": flagged_detectors, "detector_types": flagged_detectors,
            "n_facts_p1": len(r1.get("facts", [])), "n_facts_p2": len(r2.get("facts", [])),
            "n_blocks": ex["n_blocks"], "n_candidate_pairs": len(pairs),
            "extraction_latency_p1": r1.get("latency", 0.0), "extraction_latency_p2": r2.get("latency", 0.0),
            "cached_p1": r1.get("cached", False), "cached_p2": r2.get("cached", False),
            "p1_ok": r1.get("parse_ok", False), "p2_ok": r2.get("parse_ok", False),
            "nli_time": ex["nli_time"],
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
        for r in preds_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "details.jsonl"), "w", encoding="utf-8") as f:
        for r in details_rows:
            f.write(json.dumps(r) + "\n")

    print_stats(details_rows, f"{args.variant}_{args.blocking}_{args.split}")


if __name__ == "__main__":
    main()

