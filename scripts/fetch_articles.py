import os, sys, json, time, re
from datetime import datetime, timezone
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIKI_API = "https://en.wikipedia.org/w/api.php"
INPUT_PATH = os.path.join("data", "positives_split.jsonl")
ARTICLES_DIR = os.path.join("data", "articles")
REPORT_PATH = os.path.join("reports", "fetch_report.txt")
lines = []

def log(t=""):
    print(t)
    lines.append(t)

def cut_heading(text):
    pat = r"(?im)^\s*={2,}\s*(?:References|External links|See also|Notes|Further reading|Bibliography)\s*={2,}\s*$"
    m = re.search(pat, text)
    return text[: m.start()].rstrip() if m else text.rstrip()

def fetch_title(title, ua):
    params = {
        "action": "query", "prop": "extracts|revisions", "explaintext": "1",
        "rvprop": "ids|timestamp", "redirects": "1", "format": "json", "titles": title
    }
    for attempt in range(1, 3):
        try:
            r = requests.get(WIKI_API, params=params, headers={"User-Agent": ua}, timeout=15)
            if r.status_code != 200:
                if attempt == 2: return False, f"HTTP {r.status_code}", None
                time.sleep(1); continue
            pages = r.json().get("query", {}).get("pages", {})
            if not pages: return False, "No pages returned", None
            pid = list(pages.keys())[0]
            if str(pid) == "-1" or "missing" in pages[pid]:
                return False, f"Article '{title}' not found (missing)", None
            p = pages[pid]
            revs = p.get("revisions", [])
            return True, None, {
                "final_title": p.get("title", title),
                "revid": revs[0].get("revid") if revs else None,
                "revision_timestamp": revs[0].get("timestamp") if revs else None,
                "text": cut_heading(p.get("extract", ""))
            }
        except Exception as e:
            if attempt == 2: return False, str(e), None
            time.sleep(1)
    return False, "Max attempts exceeded", None

def main():
    contact = os.environ.get("WIKI_CONTACT", "").strip()
    if not contact:
        print("Error: WIKI_CONTACT environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    ua = f"ConflictLensResearch/0.1 (student research; contact {contact})"

    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.", file=sys.stderr); sys.exit(1)
    os.makedirs(ARTICLES_DIR, exist_ok=True); os.makedirs("reports", exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    target = {r["WikipediaArticleTitle"]: r["split"] for r in records if r.get("stratum") == "two_passage" and r.get("split") in ("dev", "test")}
    sorted_titles = sorted(target.keys())
    log(f"Distinct WikipediaArticleTitle count (dev & test two_passage): {len(sorted_titles)}")

    fetched, failed, char_counts = 0, [], {"dev": 0, "test": 0}

    for idx, title in enumerate(sorted_titles, start=1):
        split = target[title]
        out_file = os.path.join(ARTICLES_DIR, f"{idx:03d}.json")
        rec = None
        if os.path.exists(out_file):
            try:
                with open(out_file, "r", encoding="utf-8") as f_in:
                    ex = json.load(f_in)
                if ex.get("requested_title") == title and ex.get("text"):
                    rec = ex
            except Exception: pass

        if rec is None:
            ok, err, res = fetch_title(title, ua)
            if ok:
                rec = {
                    "requested_title": title, "final_title": res["final_title"],
                    "revid": res["revid"], "revision_timestamp": res["revision_timestamp"],
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "split": split, "text": res["text"]
                }
                with open(out_file, "w", encoding="utf-8") as f_out:
                    json.dump(rec, f_out, indent=2, ensure_ascii=False)
            else:
                failed.append((title, err))
            if idx < len(sorted_titles):
                time.sleep(1)

        if rec:
            fetched += 1
            tlen = len(rec["text"])
            char_counts[split] += tlen
            print(f"[{idx}/{len(sorted_titles)}] OK '{title}' ({split}, {tlen} chars)")
        else:
            print(f"[{idx}/{len(sorted_titles)}] FAILED '{title}'")

    log("\n" + "=" * 55 + "\nFETCH REPORT\n" + "=" * 55)
    log(f"Titles requested: {len(sorted_titles)}")
    log(f"Titles fetched:   {fetched}")
    log(f"Titles failed:    {len(failed)}")
    if failed:
        log("\nFailed titles list:")
        for t, err in failed: log(f"  - '{t}': {err}")
    else:
        log("\nAll requested titles were fetched successfully with 0 failures.")
    log("\nTotal characters per split:")
    log(f"  - dev:   {char_counts['dev']} characters")
    log(f"  - test:  {char_counts['test']} characters")
    log(f"  - Total: {char_counts['dev'] + char_counts['test']} characters")

    with open(REPORT_PATH, "w", encoding="utf-8") as f_rep:
        f_rep.write("\n".join(lines) + "\n")
    log(f"\nReport saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
