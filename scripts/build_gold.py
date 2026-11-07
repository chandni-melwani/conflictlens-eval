import os, sys, json, re, hashlib
from collections import defaultdict, Counter
import numpy as np
from transformers import AutoTokenizer

for k, v in [("CUDA_VISIBLE_DEVICES", ""), ("HF_HUB_OFFLINE", "1"), ("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"), ("TOKENIZERS_PARALLELISM", "false")]: os.environ[k] = v

def normalize(s): return " ".join(s.lower().split()) if s else ""

def map_type_group(type_I):
    if not type_I: return "unknown"
    if type_I == "(PhraseLevel) Entity - Date/time": return "date_time"
    return "number" if type_I == "(PhraseLevel) Entity - Number" else "other"

def check_constraints(items):
    dev_arts, test_arts = {x["article"] for x in items if x["split"] == "dev"}, {x["article"] for x in items if x["split"] == "test"}
    if dev_arts & test_arts: raise ValueError(f"Article overlap between dev and test: {dev_arts & test_arts}")
    seen_pairs = {}
    for x in items:
        k1, k2 = (normalize(x["passage1"]), normalize(x["passage2"])), (normalize(x["passage2"]), normalize(x["passage1"]))
        if k1 in seen_pairs or k2 in seen_pairs:
            raise ValueError(f"Duplicate passage pair: {x['id']}")
        seen_pairs[k1] = x["id"]
    for art in dev_arts | test_arts:
        pos_p = {normalize(p) for x in items if x["article"] == art and x["label"] == 1 for p in (x["passage1"], x["passage2"])}
        neg_p = {normalize(p) for x in items if x["article"] == art and x["label"] == 0 for p in (x["passage1"], x["passage2"])}
        if pos_p & neg_p: raise ValueError(f"Negative passage equals positive passage in article '{art}'")
    for x in items:
        if not x.get("passage1", "").strip() or not x.get("passage2", "").strip(): raise ValueError(f"Item {x.get('id')} has empty passage")

def select_negatives(candidates, labels_by_id, exclude_ids, quotas={"dev": 25, "test": 75}, tok=None):
    selected, excl = [], {str(x).strip() for x in exclude_ids}
    for sp in ["dev", "test"]:
        sp_c = sorted([c for c in candidates if c.get("split") == sp], key=lambda x: x.get("order_index", 0))
        taken = []
        for c in sp_c:
            cid = str(c.get("cand_id"))
            if cid in excl or not labels_by_id.get(cid) or labels_by_id[cid].get("label") != "COMPATIBLE": continue
            if len(taken) < quotas.get(sp, 0):
                lbl, ed = labels_by_id[cid], bool(labels_by_id[cid].get("edited"))
                p1 = lbl["edited_text"]["passage1"] if ed and lbl.get("edited_text") else c["passage1"]
                p2 = lbl["edited_text"]["passage2"] if ed and lbl.get("edited_text") else c["passage2"]
                pt = len(tok(p1, p2, truncation=False)["input_ids"]) if tok else 0
                taken.append({"id": f"neg_{c['cand_id']}", "split": sp, "label": 0, "article": c["article_title"],
                              "passage1": p1, "passage2": p2, "pair_tokens": pt, "contradictType": None,
                              "edited": ed, "both_digit": bool(re.search(r"\d", p1) and re.search(r"\d", p2))})
        selected.extend(taken)
    return selected

def build_gold():
    try: tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base", local_files_only=True)
    except Exception: tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
    cand_lines = [json.loads(l) for l in open("data/negative_candidates.jsonl", encoding="utf-8")]
    lbl_lines = [json.loads(l) for l in open("data/negative_labels.jsonl", encoding="utf-8")]
    cand_fields, lbl_fields = list(cand_lines[0].keys()), list(lbl_lines[0].keys())
    excl = {l.strip() for l in open("data/negative_exclude.txt", encoding="utf-8") if l.strip()} if os.path.exists("data/negative_exclude.txt") else set()
    negatives = select_negatives(cand_lines, {str(l["cand_id"]): l for l in lbl_lines}, excl, tok=tok)

    pos_all = [json.loads(l) for l in open("data/positives_split.jsonl", encoding="utf-8")]
    wiki_lookup = {}
    if os.path.exists("hf_cache/WikiContradict_dataset.json"):
        for r in json.load(open("hf_cache/WikiContradict_dataset.json", encoding="utf-8")):
            t, pa, pb = normalize(r.get("title")), normalize(r.get("paragraphA_information_standalone")), normalize(r.get("paragraphB_information_standalone"))
            v = {"type_I": r.get("Contradict type I"), "type_II": r.get("Contradict type II"), "type_III": r.get("Contradict type III")}
            wiki_lookup[(t, pa, pb)] = v; wiki_lookup[(t, pb, pa)] = v

    def make_pos_item(p):
        p1, p2, art = p["context1"], p["context2"], p["WikipediaArticleTitle"]
        m = wiki_lookup.get((normalize(art), normalize(p1), normalize(p2)))
        t1 = m["type_I"] if m else None
        return {"id": f"pos_{p['pair_id']}", "split": p["split"], "label": 1, "article": art, "passage1": p1, "passage2": p2,
                "pair_tokens": len(tok(p1, p2, truncation=False)["input_ids"]), "contradictType": p.get("contradictType"),
                "both_digit": bool(re.search(r"\d", p1) and re.search(r"\d", p2)), "type_I": t1, "type_II": m["type_II"] if m else None,
                "type_III": m["type_III"] if m else None, "type_group": map_type_group(t1)}

    pos_main = [make_pos_item(p) for p in pos_all if p.get("stratum") == "two_passage" and p.get("split") in ("dev", "test")]
    gold = pos_main + negatives
    check_constraints(gold)
    identical = [make_pos_item(p) for p in pos_all if p.get("stratum") == "same_passage"]

    os.makedirs("data", exist_ok=True)
    with open("data/gold.jsonl", "w", encoding="utf-8") as f:
        for x in gold: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open("data/gold_identical.jsonl", "w", encoding="utf-8") as f:
        for x in identical: f.write(json.dumps(x, ensure_ascii=False) + "\n")

    h_gold = hashlib.sha256(open("data/gold.jsonl", "rb").read()).hexdigest()
    h_ident = hashlib.sha256(open("data/gold_identical.jsonl", "rb").read()).hexdigest()

    lines = ["=" * 70, "GOLD DATASET REPORT", "=" * 70, f"Candidate fields: {cand_fields}\nLabel fields: {lbl_fields}", f"Excluded IDs ({len(excl)}): {sorted(list(excl))}"]
    for sp in ["dev", "test"]:
        sp_items = [x for x in gold if x["split"] == sp]
        arts = {x["article"] for x in sp_items}
        p_cnt, n_cnt = sum(1 for x in sp_items if x["label"] == 1), sum(1 for x in sp_items if x["label"] == 0)
        lines.append(f"\nSplit '{sp}': Total={len(sp_items)} (Positives={p_cnt}, Negatives={n_cnt}), Articles={len(arts)}")
        for lbl, name in [(1, "positives"), (0, "negatives")]:
            sub = [x for x in sp_items if x["label"] == lbl]
            if sub:
                toks = [x["pair_tokens"] for x in sub]
                dig, ed_sh = sum(1 for x in sub if x["both_digit"]) / len(sub), (sum(1 for x in sub if x.get("edited")) / len(sub)) if lbl == 0 else 0.0
                lines.append(f"  {name.capitalize()} (n={len(sub)}): Tokens median={np.median(toks):.1f}, p5={np.percentile(toks, 5):.1f}, p95={np.percentile(toks, 95):.1f} | Digit share={dig:.3f} | Edited share={ed_sh:.3f}")

    c_cnt, x_cnt, u_cnt = sum(1 for l in lbl_lines if l.get("label") == "COMPATIBLE"), sum(1 for l in lbl_lines if l.get("label") == "CONTRADICT"), sum(1 for l in lbl_lines if l.get("label") == "UNSURE")
    lines.extend([f"\nNegatives Labels Summary (all {len(lbl_lines)} labels):", f"  COMPATIBLE: {c_cnt}, CONTRADICT: {x_cnt}, UNSURE: {u_cnt}", f"  Edited items count: {sum(1 for l in lbl_lines if l.get('edited'))}"])
    for lb in ["COMPATIBLE", "CONTRADICT", "UNSURE"]:
        secs = [l["seconds_spent"] for l in lbl_lines if l.get("label") == lb and "seconds_spent" in l]
        lines.append(f"  {lb} seconds_spent: median={(np.median(secs) if secs else 0.0):.2f}s, mean={(np.mean(secs) if secs else 0.0):.2f}s (n={len(secs)})")

    lines.append("\nPart B Positives Type Annotation:")
    unmatched_ids = []
    for sp in ["dev", "test"]:
        sp_pos = [x for x in pos_main if x["split"] == sp]
        unmatched = [x for x in sp_pos if x["type_group"] == "unknown"]
        unmatched_ids.extend([x["id"] for x in unmatched])
        lines.append(f"  Split '{sp}': matched={len(sp_pos) - len(unmatched)}, unmatched={len(unmatched)} | type_groups={dict(Counter(x['type_group'] for x in sp_pos))}")
    lines.extend([f"  Unmatched positive IDs (up to 10): {unmatched_ids[:10]}", f"\nSHA-256 data/gold.jsonl: {h_gold}", f"SHA-256 data/gold_identical.jsonl: {h_ident}"])

    report_text = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/gold_report.txt", "w", encoding="utf-8") as f: f.write(report_text + "\n")
    print(report_text)

if __name__ == "__main__":
    build_gold()
