import os, sys, glob, json, re, random, shutil, math, argparse
from collections import defaultdict
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARTICLES_DIR = os.path.join("data", "articles")
POSITIVES_PATH = os.path.join("data", "positives_split.jsonl")
OUT_CANDIDATES = os.path.join("data", "negative_candidates.jsonl")
LABELS_PATH = os.path.join("data", "negative_labels.jsonl")
CAND_R1_PATH = os.path.join("data", "negative_candidates_round1.jsonl")
LABELS_R1_PATH = os.path.join("data", "negative_labels_round1.jsonl")
REPORT_R1_PATH = os.path.join("reports", "candidates_report.txt")
REPORT_R2_PATH = os.path.join("reports", "candidates_round2_report.txt")
TOKENIZER_ID, SEED = "cross-encoder/nli-deberta-v3-base", 42

STOPWORDS = set("about above across after again against almost along already also although always among another around became because become been before being between both came come could does doing done down during each either enough even every first former found from further gave given good great have having here however include including into just know known last later least less like made make many might more most much must name never often only other over part said same should since some still such take than that their them then there these they this those through took under until upon used using very went were what when where which while with within without would your".split())

def get_content_words(s):
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", s) if len(w) >= 4 and w.lower() not in STOPWORDS}

def clean_sentence(s):
    if "==" in s or "\n" in s or "\r" in s or "|" in s:
        return None
    s = re.sub(r"\[\d+\]", "", s).strip()
    if "==" in s or "\n" in s or "\r" in s or "|" in s:
        return None
    words = s.split()
    if len(words) < 6 or len(words) > 80:
        return None
    s_low = s.lower()
    if any(dp in s_low for dp in ["[citation needed", "[dubious", "[clarification", "contradict", "inconsistent"]):
        return None
    return s

def is_affected(cand):
    p1 = cand.get("passage1", "")
    p2 = cand.get("passage2", "")
    return ("==" in p1 or "\n" in p1 or "\r" in p1 or "==" in p2 or "\n" in p2 or "\r" in p2)

def archive_step():
    r1_cand_exists = os.path.exists(CAND_R1_PATH)
    r1_labels_exists = os.path.exists(LABELS_R1_PATH)
    if r1_cand_exists or r1_labels_exists:
        if r1_cand_exists and r1_labels_exists:
            print(f"Archive files already exist: {CAND_R1_PATH}, {LABELS_R1_PATH}. Working on existing copies.")
            return
        print(f"Error: Archive target already exists ({CAND_R1_PATH} or {LABELS_R1_PATH}). Stop.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(OUT_CANDIDATES) or not os.path.exists(LABELS_PATH):
        print(f"Error: Source files missing to archive.", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(OUT_CANDIDATES, CAND_R1_PATH)
    shutil.copy2(LABELS_PATH, LABELS_R1_PATH)
    print(f"Archived {OUT_CANDIDATES} -> {CAND_R1_PATH}")
    print(f"Archived {LABELS_PATH} -> {LABELS_R1_PATH}")

def run_round2():
    archive_step()
    with open(CAND_R1_PATH, "r", encoding="utf-8") as f:
        r1_cands = [json.loads(l) for l in f]
    with open(LABELS_R1_PATH, "r", encoding="utf-8") as f:
        r1_labels = [json.loads(l) for l in f]

    for c in r1_cands:
        c["is_affected"] = is_affected(c)
        c["is_clean"] = not c["is_affected"]
    r1_by_id = {c["cand_id"]: c for c in r1_cands}

    r1_lbl_stats = {
        sp: {"clean": defaultdict(int), "affected": defaultdict(int)}
        for sp in ["dev", "test"]
    }
    r1_clean_comp_art = defaultdict(int)
    for lb in r1_labels:
        c = r1_by_id.get(lb["cand_id"])
        if not c:
            continue
        sp = c["split"]
        cat = "affected" if c["is_affected"] else "clean"
        r1_lbl_stats[sp][cat][lb["label"]] += 1
        r1_lbl_stats[sp][cat]["total"] += 1
        if c["is_clean"] and lb["label"] == "COMPATIBLE":
            r1_clean_comp_art[c["article_title"]] += 1

    quotas = {"dev": 25, "test": 75}
    shortfall = {sp: max(0, quotas[sp] - r1_lbl_stats[sp]["clean"]["COMPATIBLE"]) for sp in ["dev", "test"]}
    target_k = {sp: math.ceil(1.5 * shortfall[sp]) for sp in ["dev", "test"]}

    print("\n--- Shortfall Calculation ---")
    for sp in ["dev", "test"]:
        print(f"Split '{sp}': quota={quotas[sp]}, kept clean COMPATIBLE={r1_lbl_stats[sp]['clean']['COMPATIBLE']}, shortfall={shortfall[sp]}, generating {target_k[sp]} round-2 candidates")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    with open(POSITIVES_PATH, "r", encoding="utf-8") as f:
        positives = [json.loads(l) for l in f if json.loads(l).get("stratum") == "two_passage" and json.loads(l).get("split") in ("dev", "test")]

    pos_by_split, pos_by_art = defaultdict(list), defaultdict(list)
    for p in positives:
        pos_by_split[p["split"]].append(p)
        pos_by_art[p["WikipediaArticleTitle"]].append(p)

    split_stats = {}
    for sp in ["dev", "test"]:
        sub = pos_by_split[sp]
        lens = [len(tokenizer(p["context1"], p["context2"], truncation=False)["input_ids"]) for p in sub]
        f_dig = sum(1 for p in sub if re.search(r"\d", p["context1"]) and re.search(r"\d", p["context2"])) / len(sub)
        split_stats[sp] = {"p5": float(np.percentile(lens, 5)), "p95": float(np.percentile(lens, 95)), "f_digit": f_dig}

    r1_norm_passages = [
        " ".join(c[k].lower().split())
        for c in r1_cands for k in ("passage1", "passage2")
    ]

    articles = []
    for af in sorted(glob.glob(os.path.join(ARTICLES_DIR, "*.json"))):
        with open(af, "r", encoding="utf-8") as f:
            articles.append(json.load(f))

    rng_order = random.Random(SEED)
    art_pairs = defaultdict(list)

    for art in articles:
        split, title = art["split"], art["requested_title"]
        pos_pass = [p["context1"] for p in pos_by_art[title]] + [p["context2"] for p in pos_by_art[title]]
        pos_words = [set(re.findall(r"\b\w+\b", p.lower())) for p in pos_pass]

        sents = []
        for line in art["text"].splitlines():
            if re.match(r"^\s*=+.*?=+\s*$", line):
                continue
            for raw in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line):
                cs = clean_sentence(raw)
                if not cs:
                    continue
                cs_low = cs.lower()
                cs_w = set(re.findall(r"\b\w+\b", cs_low))
                if any(cs_low in pt.lower() or (cs_w and len(cs_w & pw) / len(cs_w) > 0.5) for pt, pw in zip(pos_pass, pos_words)):
                    continue
                cs_norm = " ".join(cs_low.split())
                if any(cs_norm in p_norm or p_norm in cs_norm for p_norm in r1_norm_passages):
                    continue
                sents.append(cs)

        cw_list = [get_content_words(s) for s in sents]
        pairs = []
        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                shared = sorted(list(cw_list[i] & cw_list[j]))
                if len(shared) >= 2:
                    p1, p2 = (sents[i], sents[j]) if rng_order.random() < 0.5 else (sents[j], sents[i])
                    tok_len = len(tokenizer(p1, p2, truncation=False)["input_ids"])
                    if split_stats[split]["p5"] <= tok_len <= split_stats[split]["p95"]:
                        pairs.append({
                            "article_title": title, "revid": art.get("revid"), "passage1": p1, "passage2": p2,
                            "pair_tokens": tok_len, "both_digit": bool(re.search(r"\d", p1) and re.search(r"\d", p2)),
                            "shared_words": shared, "split": split
                        })
        random.Random(SEED + hash(title) % 10000).shuffle(pairs)
        art_pairs[title] = pairs

    final_r2 = []
    r2_art_counts = defaultdict(int)
    for sp in ["dev", "test"]:
        tk, f_target = target_k[sp], split_stats[sp]["f_digit"]
        sp_arts = sorted([a["requested_title"] for a in articles if a["split"] == sp])
        queues = {a: {True: [p for p in art_pairs[a] if p["both_digit"]], False: [p for p in art_pairs[a] if not p["both_digit"]]} for a in sp_arts}

        sp_selected = []
        while len(sp_selected) < tk:
            added = 0
            for a in sp_arts:
                if len(sp_selected) >= tk:
                    break
                if r2_art_counts[a] + r1_clean_comp_art[a] >= 2:
                    continue
                des_dig = (sum(1 for x in sp_selected if x["both_digit"]) < round(f_target * (len(sp_selected) + 1)))
                pick = queues[a][des_dig].pop(0) if queues[a][des_dig] else (queues[a][not des_dig].pop(0) if queues[a][not des_dig] else None)
                if pick:
                    sp_selected.append(pick)
                    r2_art_counts[a] += 1
                    added += 1
            if added == 0:
                break
        final_r2.extend(sp_selected)

    max_order_r1 = max(c["order_index"] for c in r1_cands) if r1_cands else 0
    for idx, cand in enumerate(final_r2, start=1):
        cand["cand_id"] = f"r2_{idx}"
        cand["order_index"] = max_order_r1 + idx

    # Step 4: merge
    r1_clean_cands = [c for c in r1_cands if c["is_clean"]]
    merged_cands = []
    keys_to_keep = ["article_title", "revid", "passage1", "passage2", "pair_tokens", "both_digit", "shared_words", "split", "cand_id", "order_index"]
    for c in r1_clean_cands:
        merged_cands.append({k: c[k] for k in keys_to_keep})
    for c in final_r2:
        merged_cands.append({k: c[k] for k in keys_to_keep})

    with open(OUT_CANDIDATES, "w", encoding="utf-8") as f:
        for c in merged_cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    clean_cand_ids = {c["cand_id"] for c in r1_clean_cands}
    merged_labels = [lb for lb in r1_labels if lb["cand_id"] in clean_cand_ids]
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for lb in merged_labels:
            f.write(json.dumps(lb, ensure_ascii=False) + "\n")

    # Check
    bad_count = sum(1 for c in final_r2 for p in (c["passage1"], c["passage2"]) if "==" in p or "\n" in p or "\r" in p or "|" in p)

    # Report
    rep = []
    rep.append("=" * 65 + "\nNEGATIVE CANDIDATES ROUND 2 REPORT\n" + "=" * 65)
    rep.append(f"\nRound-1 Candidates:\n  Total:    {len(r1_cands)}\n  Affected: {sum(1 for c in r1_cands if c['is_affected'])}\n  Clean:    {len(r1_clean_cands)}")
    rep.append("\nRound-1 Labels:")
    for sp in ["dev", "test"]:
        cl = r1_lbl_stats[sp]["clean"]
        af = r1_lbl_stats[sp]["affected"]
        rep.append(f"  Split '{sp}':")
        rep.append(f"    Clean:    COMPATIBLE={cl['COMPATIBLE']}, CONTRADICT={cl['CONTRADICT']}, UNSURE={cl['UNSURE']} (total: {cl['total']})")
        rep.append(f"    Affected: COMPATIBLE={af['COMPATIBLE']}, CONTRADICT={af['CONTRADICT']}, UNSURE={af['UNSURE']} (total: {af['total']})")
    rep.append("\nQuotas, Shortfall, and Round-2 Generation:")
    for sp in ["dev", "test"]:
        rep.append(f"  Split '{sp}': quota={quotas[sp]}, kept clean COMPATIBLE={r1_lbl_stats[sp]['clean']['COMPATIBLE']}, shortfall={shortfall[sp]}, round-2 generated={sum(1 for c in final_r2 if c['split'] == sp)}")
    rep.append(f"\nCheck:\n  Number of round-2 candidates containing '==', a newline or '|': {bad_count}")
    rep.append("\nRound-2 Statistics vs Positives:")
    for sp in ["dev", "test"]:
        sub = [c for c in final_r2 if c["split"] == sp]
        cov_arts = set(c["article_title"] for c in sub)
        c_digit = sum(1 for c in sub if c["both_digit"]) / len(sub) if sub else 0.0
        p_digit = split_stats[sp]["f_digit"]
        c_lens = [c["pair_tokens"] for c in sub] if sub else [0]
        pos_lens = [len(tokenizer(p["context1"], p["context2"], truncation=False)["input_ids"]) for p in pos_by_split[sp]]
        rep.append(f"\nSplit '{sp}':")
        rep.append(f"  Candidates generated: {len(sub)}")
        rep.append(f"  Articles covered:     {len(cov_arts)}")
        rep.append(f"  Digit share:          round 2 {c_digit:.3f} vs positives {p_digit:.3f}")
        rep.append(f"  Pair tokens (r2):     median={float(np.median(c_lens)):.1f}, 5th={float(np.percentile(c_lens, 5)):.1f}, 95th={float(np.percentile(c_lens, 95)):.1f}")
        rep.append(f"  Pair tokens (pos):    median={float(np.median(pos_lens)):.1f}, 5th={float(np.percentile(pos_lens, 5)):.1f}, 95th={float(np.percentile(pos_lens, 95)):.1f}")

    rep_text = "\n".join(rep)
    print("\n" + rep_text)
    with open(REPORT_R2_PATH, "w", encoding="utf-8") as f:
        f.write(rep_text + "\n")
    print(f"\nReport saved to {REPORT_R2_PATH}")
    print(f"Merged candidates saved to {OUT_CANDIDATES}")
    print(f"Merged labels saved to {LABELS_PATH}")

def run_round1():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    with open(POSITIVES_PATH, "r", encoding="utf-8") as f:
        positives = [json.loads(line) for line in f if json.loads(line).get("stratum") == "two_passage" and json.loads(line).get("split") in ("dev", "test")]

    pos_by_split, pos_by_art = defaultdict(list), defaultdict(list)
    for p in positives:
        pos_by_split[p["split"]].append(p)
        pos_by_art[p["WikipediaArticleTitle"]].append(p)

    split_stats = {}
    for sp in ["dev", "test"]:
        sub = pos_by_split[sp]
        lens = [len(tokenizer(p["context1"], p["context2"], truncation=False)["input_ids"]) for p in sub]
        f_dig = sum(1 for p in sub if re.search(r"\d", p["context1"]) and re.search(r"\d", p["context2"])) / len(sub)
        split_stats[sp] = {"p5": float(np.percentile(lens, 5)), "p95": float(np.percentile(lens, 95)), "f_digit": f_dig}

    articles = []
    for af in sorted(glob.glob(os.path.join(ARTICLES_DIR, "*.json"))):
        with open(af, "r", encoding="utf-8") as f:
            articles.append(json.load(f))

    rng_order = random.Random(SEED)
    art_pairs = defaultdict(list)

    for art in articles:
        split, title = art["split"], art["requested_title"]
        pos_passages = [p["context1"] for p in pos_by_art[title]] + [p["context2"] for p in pos_by_art[title]]
        pos_word_sets = [set(re.findall(r"\b\w+\b", p.lower())) for p in pos_passages]

        sents = []
        for raw in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", art["text"]):
            cs = clean_sentence(raw)
            if not cs:
                continue
            cs_low, cs_words = cs.lower(), set(re.findall(r"\b\w+\b", cs.lower()))
            if not any(cs_low in pt.lower() or (cs_words and len(cs_words & pw) / len(cs_words) > 0.5) for pt, pw in zip(pos_passages, pos_word_sets)):
                sents.append(cs)

        cw_list = [get_content_words(s) for s in sents]
        pairs = []
        for i in range(len(sents)):
            for j in range(i + 1, len(sents)):
                shared = sorted(list(cw_list[i] & cw_list[j]))
                if len(shared) >= 2:
                    p1, p2 = (sents[i], sents[j]) if rng_order.random() < 0.5 else (sents[j], sents[i])
                    tok_len = len(tokenizer(p1, p2, truncation=False)["input_ids"])
                    if split_stats[split]["p5"] <= tok_len <= split_stats[split]["p95"]:
                        pairs.append({
                            "article_title": title, "revid": art.get("revid"), "passage1": p1, "passage2": p2,
                            "pair_tokens": tok_len, "both_digit": bool(re.search(r"\d", p1) and re.search(r"\d", p2)),
                            "shared_words": shared, "split": split
                        })
        random.Random(SEED + hash(title) % 10000).shuffle(pairs)
        art_pairs[title] = pairs

    final_selected, quotas = [], {"dev": 35, "test": 100}
    for sp in ["dev", "test"]:
        target_k, f_target = quotas[sp], split_stats[sp]["f_digit"]
        sp_arts = sorted([a["requested_title"] for a in articles if a["split"] == sp])
        queues = {a: {True: [p for p in art_pairs[a] if p["both_digit"]], False: [p for p in art_pairs[a] if not p["both_digit"]]} for a in sp_arts}

        sp_selected = []
        while len(sp_selected) < target_k:
            added = 0
            for a in sp_arts:
                if len(sp_selected) >= target_k:
                    break
                desired_digit = (sum(1 for x in sp_selected if x["both_digit"]) < round(f_target * (len(sp_selected) + 1)))
                pick = queues[a][desired_digit].pop(0) if queues[a][desired_digit] else (queues[a][not desired_digit].pop(0) if queues[a][not desired_digit] else None)
                if pick:
                    sp_selected.append(pick)
                    added += 1
            if added == 0:
                break
        final_selected.extend(sp_selected)

    for idx, cand in enumerate(final_selected, start=1):
        cand["cand_id"] = cand["order_index"] = idx

    with open(OUT_CANDIDATES, "w", encoding="utf-8") as f:
        for c in final_selected:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lines = ["=" * 65, "NEGATIVE CANDIDATES REPORT", "=" * 65]
    for sp in ["dev", "test"]:
        sub = [c for c in final_selected if c["split"] == sp]
        cov_arts = set(c["article_title"] for c in sub)
        shortfall = max(0, quotas[sp] - len(sub))
        c_digit = sum(1 for c in sub if c["both_digit"]) / len(sub) if sub else 0.0
        p_digit = split_stats[sp]["f_digit"]
        c_lens = [c["pair_tokens"] for c in sub] if sub else [0]
        pos_lens = [len(tokenizer(p["context1"], p["context2"], truncation=False)["input_ids"]) for p in pos_by_split[sp]]

        lines.append(f"\nSplit '{sp}':")
        lines.append(f"  Candidates generated: {len(sub)} (quota: {quotas[sp]}, shortfall: {shortfall})")
        lines.append(f"  Articles covered:     {len(cov_arts)}")
        lines.append(f"  Digit share:          candidates {c_digit:.3f} vs positives {p_digit:.3f}")
        lines.append(f"  Pair tokens (cand):   median={float(np.median(c_lens)):.1f}, 5th={float(np.percentile(c_lens, 5)):.1f}, 95th={float(np.percentile(c_lens, 95)):.1f}")
        lines.append(f"  Pair tokens (pos):    median={float(np.median(pos_lens)):.1f}, 5th={float(np.percentile(pos_lens, 5)):.1f}, 95th={float(np.percentile(pos_lens, 95)):.1f}")

    with open(REPORT_R1_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport saved to {REPORT_R1_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Propose negative pairs")
    parser.add_argument("--round", type=int, default=2, choices=[1, 2], help="Generation round (1 or 2)")
    args = parser.parse_args()
    if args.round == 1:
        run_round1()
    else:
        run_round2()

if __name__ == "__main__":
    main()
