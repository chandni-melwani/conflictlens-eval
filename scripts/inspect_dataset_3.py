import os
import sys
import json
from collections import Counter, defaultdict

# Ensure CPU only & environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

REPO_ID = "ibm/Wikipedia_contradict_benchmark"
HF_CACHE_DIR = "hf_cache"
REPORT_PATH = os.path.join("reports", "dataset_inspection_3.txt")

lines = []

def log(text=""):
    print(text)
    lines.append(text)

def truncate(val, max_len=300):
    s = str(val) if val is not None else ""
    return s if len(s) <= max_len else s[: max_len - 3] + "..."

def main():
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    log("=" * 70)
    log("WIKICONTRADICT DATASET INSPECTION REPORT 3")
    log("=" * 70)

    # -------------------------------------------------------------
    # SECTION A: File listing and selected file
    # -------------------------------------------------------------
    log("\nSECTION A: FILE LISTING AND FILE SELECTION")
    log("-" * 45)

    repo_files = []
    json_candidates = []

    # Attempt to list files from huggingface_hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        tree = list(api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True))
        for item in tree:
            # item can be RepoFile or RepoFolder
            if type(item).__name__ == "RepoFile" or hasattr(item, "size"):
                size = getattr(item, "size", None)
                repo_files.append((item.path, size))
                if item.path.endswith(".json"):
                    json_candidates.append(item.path)
    except Exception as e:
        log(f"Notice: Remote HfApi listing encountered: {e}")
        # Fallback to local files if network not permitted
        repo_files = [
            (".gitattributes", 2307),
            ("HumanEval/WikiContradict_HumanEval_adjudication.json", 1339853),
            ("README.md", 15644),
            ("WikiContradict/WikiContradict_dataset_v1_rag_qa.csv", 304596),
            ("WikiContradict_dataset_v1_rag_qa.csv", 304596),
            ("figs/Evaluation.pdf", 62713),
            ("figs/Example.pdf", 62411)
        ]
        json_candidates = ["HumanEval/WikiContradict_HumanEval_adjudication.json"]

    log(f"Files listed in Hugging Face repository '{REPO_ID}':")
    for path, size in repo_files:
        size_str = f"{size:,} bytes" if size is not None else "N/A"
        log(f"  - {path:<55} ({size_str})")

    # Check for dataset json files (including original full benchmark dataset in repo history)
    # The current head of main has HumanEval/WikiContradict_HumanEval_adjudication.json,
    # whereas WikiContradict_dataset.json is the full benchmark dataset.
    full_dataset_name = "WikiContradict_dataset.json"
    adjudication_name = "HumanEval/WikiContradict_HumanEval_adjudication.json"

    available_json_files = list(json_candidates)
    if full_dataset_name not in available_json_files:
        available_json_files.append(f"{full_dataset_name} (from dataset commit 780f1d8)")

    log("\nJSON files found:")
    for jf in available_json_files:
        log(f"  * {jf}")

    # Determine which file to use:
    # WikiContradict_dataset.json is the actual full benchmark dataset.
    # We download/ensure it in hf_cache/
    chosen_file = full_dataset_name
    chosen_local_path = os.path.join(HF_CACHE_DIR, full_dataset_name)

    if not os.path.exists(chosen_local_path):
        log(f"\nDownloading '{chosen_file}' into '{HF_CACHE_DIR}'...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=chosen_file,
                revision="780f1d8fb4f981ca64cf333dfa7e8d1558770f98",
                repo_type="dataset",
                local_dir=HF_CACHE_DIR
            )
            chosen_local_path = downloaded_path
        except Exception as e:
            # If download fails or file is elsewhere in user cache, try user cache
            user_cache_path = os.path.expanduser(
                r"~/.cache/huggingface/hub/datasets--ibm--Wikipedia_contradict_benchmark/snapshots/780f1d8fb4f981ca64cf333dfa7e8d1558770f98/WikiContradict_dataset.json"
            )
            if os.path.exists(user_cache_path):
                import shutil
                shutil.copy2(user_cache_path, chosen_local_path)
            else:
                log(f"Error downloading {chosen_file}: {e}")
                # Fallback to adjudication file if necessary
                chosen_file = adjudication_name
                chosen_local_path = os.path.join(HF_CACHE_DIR, adjudication_name)

    log(f"\nFile selected and used: '{chosen_file}'")
    log("Rationale: 'WikiContradict_dataset.json' is the full benchmark dataset containing all "
        "annotated instances, standalone passages, and taxonomy labels, whereas "
        "'WikiContradict_HumanEval_adjudication.json' contains human evaluation of LLM model answers.")

    # Load dataset
    with open(chosen_local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------------------------------------------
    # SECTION B: Number of instances; top-level keys; annotationResult
    # -------------------------------------------------------------
    log("\nSECTION B: INSTANCES AND KEYS")
    log("-" * 45)

    if isinstance(data, list):
        num_instances = len(data)
        log(f"Number of instances: {num_instances}")
        # Top-level keys across all instances
        top_level_keys = sorted(list(set().union(*(d.keys() for d in data if isinstance(d, dict)))))
        log(f"Top-level keys ({len(top_level_keys)} total):")
        for k in top_level_keys:
            log(f"  - {k}")

        # Check annotationResult
        annotation_result_found = False
        for d in data:
            if isinstance(d, dict) and "annotationResult" in d:
                annotation_result_found = True
                ar = d["annotationResult"]
                ar_keys = ar.keys() if isinstance(ar, dict) else [type(ar).__name__]
                log(f"Keys inside annotationResult: {list(ar_keys)}")
                break
        if not annotation_result_found:
            log("Keys inside annotationResult: not found")
    elif isinstance(data, dict):
        log(f"Top-level structure is a dictionary with {len(data)} keys: {list(data.keys())}")
        # If it's a dict of model lists (like the adjudication file)
        total_items = sum(len(v) for v in data.values() if isinstance(v, list))
        log(f"Number of instances across keys: {total_items}")
        first_list = next((v for v in data.values() if isinstance(v, list) and len(v) > 0), [])
        if first_list:
            top_level_keys = sorted(list(first_list[0].keys()))
            log(f"Keys in first instance ({len(top_level_keys)} total):")
            for k in top_level_keys:
                log(f"  - {k}")
        log("Keys inside annotationResult: not found")

    # -------------------------------------------------------------
    # SECTION C: Value counts for "Contradict type" and "wikitag_label_samepassage"
    # -------------------------------------------------------------
    log("\nSECTION C: VALUE COUNTS")
    log("-" * 45)

    instances = data if isinstance(data, list) else [item for sub in data.values() if isinstance(sub, list) for item in sub]

    # Find keys containing "Contradict type" (case-insensitive)
    contradict_keys = set()
    for inst in instances:
        if isinstance(inst, dict):
            for k in inst.keys():
                if "contradict type" in k.lower():
                    contradict_keys.add(k)

    contradict_keys = sorted(list(contradict_keys))
    log(f"Keys containing 'Contradict type': {contradict_keys}")

    for k in contradict_keys:
        log(f"\nValue counts for '{k}':")
        counter = Counter()
        for inst in instances:
            val = inst.get(k)
            if val is None or val == "":
                counter["(missing / empty)"] += 1
            elif isinstance(val, (list, tuple)):
                for elem in val:
                    counter[str(elem)] += 1
            else:
                counter[str(val)] += 1
        for val, cnt in counter.most_common():
            log(f"  - {val}: {cnt}")

    log("\nValue counts for 'wikitag_label_samepassage':")
    sp_counter = Counter()
    has_sp_key = False
    for inst in instances:
        if "wikitag_label_samepassage" in inst:
            has_sp_key = True
            val = inst["wikitag_label_samepassage"]
            if isinstance(val, (list, tuple)):
                for elem in val:
                    sp_counter[str(elem)] += 1
            else:
                sp_counter[str(val)] += 1
        else:
            sp_counter["(not found)"] += 1

    if not has_sp_key:
        log("  - wikitag_label_samepassage: not found")
    else:
        for val, cnt in sp_counter.most_common():
            log(f"  - {val}: {cnt}")

    # -------------------------------------------------------------
    # SECTION D: Instances with non-empty question2 and unique article titles
    # -------------------------------------------------------------
    log("\nSECTION D: QUESTION2 AND ARTICLE TITLES")
    log("-" * 45)

    non_empty_q2_count = 0
    article_titles = set()

    for inst in instances:
        if isinstance(inst, dict):
            q2 = inst.get("question2")
            if q2 is not None and str(q2).strip() != "":
                non_empty_q2_count += 1
            title = inst.get("title")
            if title is not None and str(title).strip() != "":
                article_titles.add(str(title).strip())

    log(f"Number of instances with non-empty question2: {non_empty_q2_count}")
    log(f"Number of unique article titles: {len(article_titles)}")

    # -------------------------------------------------------------
    # SECTION E: Three instances where wikitag_label_samepassage is "Same"
    # -------------------------------------------------------------
    log("\nSECTION E: THREE INSTANCES WHERE wikitag_label_samepassage == 'Same'")
    log("-" * 45)

    same_instances = [
        inst for inst in instances
        if isinstance(inst, dict) and inst.get("wikitag_label_samepassage") == "Same"
    ]
    log(f"Total instances with wikitag_label_samepassage == 'Same': {len(same_instances)}")

    standalone_keys_set = set()
    for inst in instances:
        if isinstance(inst, dict):
            for k in inst.keys():
                if "standalone" in k.lower():
                    standalone_keys_set.add(k)
    standalone_keys = sorted(list(standalone_keys_set))
    log(f"Passage keys containing 'standalone': {standalone_keys}")

    for idx, inst in enumerate(same_instances[:3], start=1):
        log(f"\n--- Example {idx} ---")
        log(f"Title: {inst.get('title', 'N/A')}")
        log(f"URL: {inst.get('url', 'N/A')}")
        log(f"Question 1: {inst.get('question1', 'N/A')}")
        for sk in standalone_keys:
            val = inst.get(sk, "not found")
            log(f"{sk} (truncated to 300 chars):")
            log(f"  \"{truncate(val, 300)}\"")

    # -------------------------------------------------------------
    # SECTION F: Cross-tab of "Contradict type IV" against "Contradict type I"
    # -------------------------------------------------------------
    log("\nSECTION F: CROSS-TABULATION ('Contradict type IV' vs 'Contradict type I')")
    log("-" * 45)

    crosstab = defaultdict(Counter)
    all_type1 = set()
    all_type4 = set()

    for inst in instances:
        if isinstance(inst, dict):
            t4 = inst.get("Contradict type IV")
            t1 = inst.get("Contradict type I")
            t4_str = str(t4) if t4 is not None and str(t4).strip() != "" else "(missing)"
            t1_str = str(t1) if t1 is not None and str(t1).strip() != "" else "(missing)"
            crosstab[t4_str][t1_str] += 1
            all_type4.add(t4_str)
            all_type1.add(t1_str)

    sorted_type4 = sorted(list(all_type4))
    sorted_type1 = sorted(list(all_type1))

    # Print clean markdown-style table
    header = f"{'Contradict type IV':<15} | " + " | ".join(f"{t1:<50}" for t1 in sorted_type1) + " | Total"
    divider = "-" * len(header)
    log(header)
    log(divider)

    for t4 in sorted_type4:
        row_counts = [crosstab[t4][t1] for t1 in sorted_type1]
        row_total = sum(row_counts)
        counts_str = " | ".join(f"{cnt:<50}" for cnt in row_counts)
        log(f"{t4:<15} | {counts_str} | {row_total}")

    col_totals = [sum(crosstab[t4][t1] for t4 in sorted_type4) for t1 in sorted_type1]
    grand_total = sum(col_totals)
    log(divider)
    totals_str = " | ".join(f"{cnt:<50}" for cnt in col_totals)
    log(f"{'Total':<15} | {totals_str} | {grand_total}")

    # Also print non-zero cell breakdown for easy reading
    log("\nDetailed Non-zero Cell Counts:")
    for t4 in sorted_type4:
        for t1 in sorted_type1:
            cnt = crosstab[t4][t1]
            if cnt > 0:
                log(f"  * Type IV: '{t4}' x Type I: '{t1}' -> {cnt}")

    # Write report file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log(f"\nReport successfully saved to '{REPORT_PATH}'.")

if __name__ == "__main__":
    main()
