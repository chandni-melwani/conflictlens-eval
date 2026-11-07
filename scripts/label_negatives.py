import os, sys, json, time, re
from datetime import datetime, timezone
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CAND_PATH = os.path.join("data", "negative_candidates.jsonl")
LABELS_PATH = os.path.join("data", "negative_labels.jsonl")
FINAL_PATH = os.path.join("data", "negatives_final.jsonl")
REPORT_PATH = os.path.join("reports", "negatives_report.txt")
POSITIVES_PATH = os.path.join("data", "positives_split.jsonl")
TOKENIZER_ID = "cross-encoder/nli-deberta-v3-base"
QUOTAS = {"dev": 25, "test": 75}

def finalize(candidates_by_id):
    labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [json.loads(l) for l in f]
    
    comp_by_split = {"dev": [], "test": []}
    for lb in labels:
        cid = lb["cand_id"]
        cand = candidates_by_id.get(cid)
        if cand and lb["label"] == "COMPATIBLE":
            sp = cand["split"]
            if len(comp_by_split[sp]) < QUOTAS[sp]:
                p1 = lb["edited_text"]["passage1"] if lb["edited"] else cand["passage1"]
                p2 = lb["edited_text"]["passage2"] if lb["edited"] else cand["passage2"]
                comp_by_split[sp].append({**cand, "passage1": p1, "passage2": p2, "edited": lb["edited"]})

    final_list = comp_by_split["dev"] + comp_by_split["test"]
    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        for it in final_list: f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Positives for comparison
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    with open(POSITIVES_PATH, "r", encoding="utf-8") as f:
        pos = [json.loads(l) for l in f if json.loads(l).get("stratum") == "two_passage"]

    lines = ["=" * 60, "NEGATIVES FINAL REPORT", "=" * 60]
    for sp in ["dev", "test"]:
        sp_lbs = [l for l in labels if candidates_by_id.get(l["cand_id"], {}).get("split") == sp]
        c_cnt = sum(1 for l in sp_lbs if l["label"] == "COMPATIBLE")
        x_cnt = sum(1 for l in sp_lbs if l["label"] == "CONTRADICT")
        u_cnt = sum(1 for l in sp_lbs if l["label"] == "UNSURE")
        lines.append(f"\nSplit '{sp}':")
        lines.append(f"  Labels: COMPATIBLE={c_cnt}, CONTRADICT={x_cnt}, UNSURE={u_cnt} (total: {len(sp_lbs)})")
        neg_sp = comp_by_split[sp]
        arts = set(x["article_title"] for x in neg_sp)
        lines.append(f"  Final negatives: {len(neg_sp)} across {len(arts)} articles")
        if neg_sp:
            neg_lens = [len(tok(x["passage1"], x["passage2"], truncation=False)["input_ids"]) for x in neg_sp]
            neg_dig = sum(1 for x in neg_sp if re.search(r"\d", x["passage1"]) and re.search(r"\d", x["passage2"])) / len(neg_sp)
            pos_sp = [p for p in pos if p["split"] == sp]
            pos_lens = [len(tok(p["context1"], p["context2"], truncation=False)["input_ids"]) for p in pos_sp]
            pos_dig = sum(1 for p in pos_sp if re.search(r"\d", p["context1"]) and re.search(r"\d", p["context2"])) / len(pos_sp)
            lines.append(f"  Median pair tokens: negatives {np.median(neg_lens):.1f} vs positives {np.median(pos_lens):.1f}")
            lines.append(f"  Digit share:        negatives {neg_dig:.3f} vs positives {pos_dig:.3f}")

    edited_count = sum(1 for l in labels if l.get("edited"))
    secs = [l.get("seconds_spent", 0.0) for l in labels]
    med_sec = float(np.median(secs)) if secs else 0.0
    lines.append(f"\nOverall: Edited items={edited_count}, Median seconds/label={med_sec:.1f}s")
    rep_text = "\n".join(lines)
    print(rep_text)
    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write(rep_text + "\n")
    print(f"\nFinal saved to {FINAL_PATH} and report to {REPORT_PATH}")

def main():
    if not os.path.exists(CAND_PATH):
        print(f"Error: {CAND_PATH} not found.", file=sys.stderr); sys.exit(1)
    with open(CAND_PATH, "r", encoding="utf-8") as f:
        candidates = [json.loads(l) for l in f]
    candidates.sort(key=lambda x: x["order_index"])
    cand_by_id = {c["cand_id"]: c for c in candidates}

    labeled_ids = set()
    compatible_counts = {"dev": 0, "test": 0}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for l in f:
                rec = json.loads(l)
                labeled_ids.add(rec["cand_id"])
                if rec["label"] == "COMPATIBLE" and rec["cand_id"] in cand_by_id:
                    compatible_counts[cand_by_id[rec["cand_id"]]["split"]] += 1

    queue = [c for c in candidates if c["cand_id"] not in labeled_ids]
    idx = 0
    while idx < len(queue):
        cand = queue[idx]
        sp = cand["split"]
        if compatible_counts[sp] >= QUOTAS[sp]:
            idx += 1; continue

        print("\n" + "=" * 65)
        print(f"Progress: [dev: {compatible_counts['dev']}/{QUOTAS['dev']} | test: {compatible_counts['test']}/{QUOTAS['test']}]")
        print(f"Candidate #{cand['cand_id']} (order {cand['order_index']}) | Split: {sp} | Article: {cand['article_title']}")
        print(f"Passage 1: {cand['passage1']}")
        print(f"Passage 2: {cand['passage2']}")
        print("-" * 65)

        t0 = time.time()
        try:
            cmd = input("[c]ompatible, [x] contradict, [u]nsure, [e]dit, [s]kip, [f]inalize, [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break

        if cmd == "q":
            print("Exiting without saving label."); break
        elif cmd == "f":
            finalize(cand_by_id); break
        elif cmd == "s":
            idx += 1; continue
        elif cmd in ("c", "x", "u", "e"):
            edited, p1, p2 = False, cand["passage1"], cand["passage2"]
            if cmd == "e":
                which = input("Edit passage [1], [2], or [b]oth? ").strip().lower()
                if which in ("1", "b"): p1 = input("New passage 1: ").strip() or p1
                if which in ("2", "b"): p2 = input("New passage 2: ").strip() or p2
                edited = True
                cmd = input("Label for edited pair ([c]ompatible / [x] contradict / [u]nsure): ").strip().lower()
                if cmd not in ("c", "x", "u"):
                    print("Invalid label; skipped."); idx += 1; continue

            lbl_map = {"c": "COMPATIBLE", "x": "CONTRADICT", "u": "UNSURE"}
            lbl = lbl_map[cmd]
            sec_spent = round(time.time() - t0, 2)
            rec = {
                "cand_id": cand["cand_id"], "label": lbl, "edited": edited,
                "original_text": {"passage1": cand["passage1"], "passage2": cand["passage2"]},
                "edited_text": {"passage1": p1, "passage2": p2},
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seconds_spent": sec_spent
            }
            with open(LABELS_PATH, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if lbl == "COMPATIBLE": compatible_counts[sp] += 1
            idx += 1
        else:
            print("Invalid input.")

if __name__ == "__main__":
    main()
