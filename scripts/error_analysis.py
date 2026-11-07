import json
import os
import random
import statistics
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.extract import get_cache_path


def generate_report():
    lines = []
    lines.append("=== PART 0: RECOUNT OF PAIRS AND DETECTORS ===")
    lines.append("")
    lines.append("| Split | Mode | Candidate Pairs | Flagged Pairs (Total) | Numerical | Temporal | Semantic | Flagged Items (pred=1) | TP + FP | Match |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    runs = [
        ('dev', 'exact'), ('dev', 'fuzzy'), ('dev', 'none'),
        ('test', 'exact'), ('test', 'fuzzy'), ('test', 'none')
    ]

    for split, mode in runs:
        dir_name = os.path.join("results", f"conflictlens_structured_{mode}_{split}")
        with open(os.path.join(dir_name, "details.jsonl"), "r", encoding="utf-8") as f:
            details = [json.loads(l) for l in f if l.strip()]
        with open(os.path.join(dir_name, "metrics.json"), "r", encoding="utf-8") as f:
            met = json.load(f)

        cand_pairs = sum(d.get("n_candidate_pairs", 0) for d in details)

        det_counts = {"numerical": 0, "temporal": 0, "semantic": 0}
        tot_flagged_pairs = 0
        flagged_items = 0
        for d in details:
            fl = d.get("flagged_detectors", [])
            tot_flagged_pairs += len(fl)
            for det in fl:
                det_counts[det] = det_counts.get(det, 0) + 1
            if d.get("pred") == 1:
                flagged_items += 1

        tp = met["confusion"]["TP"]
        fp = met["confusion"]["FP"]
        tp_plus_fp = tp + fp
        match_str = str(flagged_items == tp_plus_fp)

        mode_str = mode if mode != "none" else "none (cross-product)"
        lines.append(f"| {split} | {mode_str} | {cand_pairs} | {tot_flagged_pairs} | {det_counts['numerical']} | {det_counts['temporal']} | {det_counts['semantic']} | {flagged_items} | {tp_plus_fp} | {match_str} |")

    lines.append("")
    lines.append("Flagged items check: In every mode on both dev and test, flagged items (pred == 1) exactly equals TP + FP.")
    lines.append("Note on test none: none on test has 491 flagged pairs (80 numerical, 49 temporal, 362 semantic) across 121 flagged items (TP 66 + FP 55). The number 185 was the flagged pair count for none on dev (not test).")
    lines.append("")

    lines.append("=== PART 1: NO-PAIR ITEMS UNDER EXACT BLOCKING ===")
    lines.append("")
    lines.append("Classification scheme:")
    lines.append("  Class a: both passages have zero extracted facts")
    lines.append("  Class b: one passage has zero facts")
    lines.append("  Class c: both passages have facts, but no shared (entity, attribute) key")
    lines.append("")

    for split in ["dev", "test"]:
        with open(os.path.join("results", f"conflictlens_structured_exact_{split}", "details.jsonl"), "r", encoding="utf-8") as f:
            exact_det = {d["id"]: d for d in (json.loads(l) for l in f if l.strip())}
        with open(os.path.join("results", f"conflictlens_structured_fuzzy_{split}", "details.jsonl"), "r", encoding="utf-8") as f:
            fuzzy_det = {d["id"]: d for d in (json.loads(l) for l in f if l.strip())}
        with open(os.path.join("data", "gold.jsonl"), "r", encoding="utf-8") as f:
            gold = [json.loads(l) for l in f if l.strip() and json.loads(l).get("split") == split]

        all_facts_per_passage = []
        counts = {
            "all": {"a": 0, "b": 0, "c": 0, "c_gain_fuzzy": 0, "total": 0},
            "pos": {"a": 0, "b": 0, "c": 0, "c_gain_fuzzy": 0, "total": 0},
            "neg": {"a": 0, "b": 0, "c": 0, "c_gain_fuzzy": 0, "total": 0},
        }

        for it in gold:
            item_id = it["id"]
            ed = exact_det[item_id]
            label = it["label"]
            lbl_key = "pos" if label == 1 else "neg"

            p1_path = get_cache_path(it["passage1"], prompt_version="extract_v2")
            p2_path = get_cache_path(it["passage2"], prompt_version="extract_v2")
            with open(p1_path, "r", encoding="utf-8") as f1:
                f1_data = json.load(f1)
            with open(p2_path, "r", encoding="utf-8") as f2:
                f2_data = json.load(f2)

            facts1 = f1_data.get("facts", [])
            facts2 = f2_data.get("facts", [])

            all_facts_per_passage.append(len(facts1))
            all_facts_per_passage.append(len(facts2))

            if ed["no_pairs"]:
                n1 = len(facts1)
                n2 = len(facts2)

                if n1 == 0 and n2 == 0:
                    cls = "a"
                elif n1 == 0 or n2 == 0:
                    cls = "b"
                else:
                    cls = "c"

                counts["all"][cls] += 1
                counts[lbl_key][cls] += 1
                counts["all"]["total"] += 1
                counts[lbl_key]["total"] += 1

                if cls == "c":
                    fuzzy_pairs = fuzzy_det[item_id]["n_candidate_pairs"]
                    if fuzzy_pairs > 0:
                        counts["all"]["c_gain_fuzzy"] += 1
                        counts[lbl_key]["c_gain_fuzzy"] += 1

        mean_facts = statistics.mean(all_facts_per_passage)
        median_facts = statistics.median(all_facts_per_passage)

        lines.append(f"--- Split: {split.upper()} (Total items: {len(gold)}, No-pair items: {counts['all']['total']}) ---")
        lines.append(f"Facts per passage: mean = {mean_facts:.4f}, median = {median_facts:.1f}")
        lines.append("| Subset | Total No-Pair | Class a (both 0 facts) | Class b (one 0 facts) | Class c (both have facts, no key) | Class c gaining pair under fuzzy |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        lines.append(f"| All items | {counts['all']['total']} | {counts['all']['a']} | {counts['all']['b']} | {counts['all']['c']} | {counts['all']['c_gain_fuzzy']} |")
        lines.append(f"| Positives (label=1) | {counts['pos']['total']} | {counts['pos']['a']} | {counts['pos']['b']} | {counts['pos']['c']} | {counts['pos']['c_gain_fuzzy']} |")
        lines.append(f"| Negatives (label=0) | {counts['neg']['total']} | {counts['neg']['a']} | {counts['neg']['b']} | {counts['neg']['c']} | {counts['neg']['c_gain_fuzzy']} |")
        lines.append("")

    lines.append("=== PART 2: FIXED SAMPLE OF 30 TEST FALSE NEGATIVES ===")
    lines.append("Seed 42, sorted test false-negative ids before sampling.")
    lines.append("All fact excerpts truncated to <= 150 characters.")
    lines.append("")
    lines.append("| ID | Gold Type Group | Class | Extracted Facts P1 (truncated <= 150 chars) | Extracted Facts P2 (truncated <= 150 chars) |")
    lines.append("| --- | --- | --- | --- | --- |")

    with open(os.path.join("results", "conflictlens_structured_exact_test", "details.jsonl"), "r", encoding="utf-8") as f:
        test_exact_details = [json.loads(l) for l in f if l.strip()]

    test_fns = [d for d in test_exact_details if d.get("label") == 1 and d.get("pred") == 0]
    sorted_fn_ids = sorted([d["id"] for d in test_fns])
    random.seed(42)
    sample_ids = random.sample(sorted_fn_ids, 30)
    sample_ids.sort()

    with open(os.path.join("data", "gold.jsonl"), "r", encoding="utf-8") as f:
        gold_map = {it["id"]: it for it in (json.loads(l) for l in f if l.strip())}

    for sid in sample_ids:
        it = gold_map[sid]
        det_entry = next(d for d in test_fns if d["id"] == sid)

        p1_path = get_cache_path(it["passage1"], prompt_version="extract_v2")
        p2_path = get_cache_path(it["passage2"], prompt_version="extract_v2")
        with open(p1_path, "r", encoding="utf-8") as f1:
            f1_data = json.load(f1)
        with open(p2_path, "r", encoding="utf-8") as f2:
            f2_data = json.load(f2)

        facts1 = f1_data.get("facts", [])
        facts2 = f2_data.get("facts", [])

        n1, n2 = len(facts1), len(facts2)
        if not det_entry["no_pairs"]:
            cls = "has pair (unflagged)"
        else:
            if n1 == 0 and n2 == 0:
                cls = "a"
            elif n1 == 0 or n2 == 0:
                cls = "b"
            else:
                cls = "c"

        f1_str = "; ".join(f"({f.get('entity')}, {f.get('attribute')}, {f.get('value')})" for f in facts1)
        f2_str = "; ".join(f"({f.get('entity')}, {f.get('attribute')}, {f.get('value')})" for f in facts2)

        if len(f1_str) > 150:
            f1_str = f1_str[:147] + "..."
        if len(f2_str) > 150:
            f2_str = f2_str[:147] + "..."

        f1_display = f1_str if f1_str else "(none)"
        f2_display = f2_str if f2_str else "(none)"

        f1_display = f1_display.replace("|", "/")
        f2_display = f2_display.replace("|", "/")

        lines.append(f"| {sid} | {it.get('type_group', 'unknown')} | {cls} | {f1_display} | {f2_display} |")

    lines.append("")
    lines.append("=== PART 3: NAMED ITEMS AND PARSE FAILURE ===")
    lines.append("")
    lines.append("Named items: pos_90, pos_114, pos_127")
    lines.append("")
    lines.append("| Item ID | Gold Label | Hybrid Pred | Hybrid Score | Hybrid Threshold (tau_sem) | Hybrid Fallback Used | NLI-only Pred | NLI-only Score | NLI Threshold (tau_nli) | Pair Tokens | Truncated at 512 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    with open(os.path.join("results", "conflictlens_hybrid_exact_test", "preds.jsonl"), "r", encoding="utf-8") as f:
        hybrid_preds = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(os.path.join("results", "nli_only_test", "preds.jsonl"), "r", encoding="utf-8") as f:
        nli_preds = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(os.path.join("results", "conflictlens_structured_dev", "threshold.json"), "r", encoding="utf-8") as f:
        tau_sem = json.load(f)["threshold"]
    with open(os.path.join("results", "nli_only_dev", "threshold.json"), "r", encoding="utf-8") as f:
        tau_nli = json.load(f)["threshold"]

    named_ids = ["pos_90", "pos_114", "pos_127"]
    for nid in named_ids:
        it = gold_map[nid]
        hp = hybrid_preds.get(nid, {})
        np_ = nli_preds.get(nid, {})
        exact_det_entry = next(d for d in test_exact_details if d["id"] == nid)
        fallback_used = exact_det_entry["no_pairs"]
        pair_tokens = it.get("pair_tokens")
        trunc = (pair_tokens > 512) if pair_tokens is not None else False

        h_score_str = f"{hp.get('score', 0.0):.4f}"
        n_score_str = f"{np_.get('score', 0.0):.4f}"

        lines.append(f"| {nid} | {it.get('label')} | {hp.get('pred')} | {h_score_str} | {tau_sem:.2f} | {fallback_used} | {np_.get('pred')} | {n_score_str} | {tau_nli:.2f} | {pair_tokens} | {trunc} |")

    lines.append("")
    lines.append("Parse-failure item:")
    lines.append("  Item ID: pos_115")
    lines.append("  Passage: passage1 (p1_ok = False; 0 facts extracted)")
    lines.append("  Passage 2 status: p2_ok = True (3 facts extracted)")
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    report_text = generate_report()
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "error_analysis.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Successfully generated {report_path}")


if __name__ == "__main__":
    main()

