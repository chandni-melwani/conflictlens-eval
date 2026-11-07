import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.metrics import confusion, mcc
from scripts.nli import contradiction_score

JUDGE_PROMPT_TEMPLATE = (
    "You compare two passages from the same Wikipedia article. Decide whether they make claims that cannot "
    "both be true. Passages that discuss different things, or that add detail without conflicting, are not "
    'contradictions. Answer only with JSON: {"contradiction": true} or {"contradiction": false}.\n\n'
    "Passage 1: {p1}\n\nPassage 2: {p2}\n\nAnswer:"
)


def fill_judge_prompt(p1: str, p2: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.replace("{p1}", p1).replace("{p2}", p2)


def find_best_threshold(labels: list[int], scores: list[float], thresholds: list[float] | None = None):
    thresholds = thresholds or [round(i * 0.05, 2) for i in range(1, 20)]
    best_t, best_mcc, curve = thresholds[0], -2.0, []
    for t in sorted(thresholds):
        tp, fp, fn, tn = confusion(labels, [1 if s >= t else 0 for s in scores])
        val = mcc(tp=tp, fp=fp, fn=fn, tn=tn)
        curve.append({"threshold": t, "mcc": val})
        if val > best_mcc:
            best_mcc, best_t = val, t
    return best_t, best_mcc, curve


def call_ollama(prompt: str, base_url: str = "http://localhost:11434") -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": "qwen2.5:3b", "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0, "seed": 42, "num_ctx": 4096}}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_judge_json(raw: str):
    try:
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()) if raw else ""
        d = json.loads(t)
        if isinstance(d, dict) and isinstance(d.get("contradiction"), bool):
            return True, (1 if d["contradiction"] else 0)
    except Exception:
        pass
    return False, 0


def query_judge(p1: str, p2: str, base_url: str = "http://localhost:11434", cache_dir: str = "data/cache/judge"):
    os.makedirs(cache_dir, exist_ok=True)
    ck = hashlib.sha256(f"qwen2.5:3b_judge_v1_{p1}_{p2}".encode("utf-8")).hexdigest()
    cp = os.path.join(cache_dir, f"{ck}.json")
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d["pred"], d["parse_ok"]
    prompt = fill_judge_prompt(p1, p2)
    raw = call_ollama(prompt, base_url)
    ok, pred = parse_judge_json(raw)
    if not ok:
        raw = call_ollama(prompt + "\n\nCRITICAL: Answer ONLY with valid JSON: {\"contradiction\": true} or {\"contradiction\": false}.", base_url)
        ok, pred = parse_judge_json(raw)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump({"model": "qwen2.5:3b", "prompt_version": "judge_v1", "pred": pred if ok else 0, "parse_ok": ok, "raw": raw}, f, indent=2)
    return (pred if ok else 0), ok


def run_nli_baseline(items: list[dict], split: str, threshold_file: str):
    print(f"\n--- Running nli_only on split: {split} ({len(items)} items) ---")
    if split != "dev":
        if not os.path.exists(threshold_file):
            raise FileNotFoundError(f"Missing threshold file: {threshold_file}. Never choose threshold on {split}.")
        with open(threshold_file, "r", encoding="utf-8") as f:
            threshold = float(json.load(f)["threshold"])
        print(f"Loaded threshold: {threshold:.2f} from {threshold_file}")

    truncated = sum(1 for it in items if it.get("pair_tokens", 0) > 512)
    print(f"Items truncated (> 512 tokens): {truncated}")
    scores = [contradiction_score(it["passage1"], it["passage2"]) for it in items]
    out_dir = os.path.join("results", f"nli_only_{split}")
    os.makedirs(out_dir, exist_ok=True)
    if split == "dev":
        best_t, best_mcc, curve = find_best_threshold([it["label"] for it in items], scores)
        print(f"Chosen dev threshold: {best_t:.2f} (Dev MCC: {best_mcc:.4f})")
        with open(threshold_file, "w", encoding="utf-8") as f:
            json.dump({"threshold": best_t, "mcc": best_mcc, "curve": curve}, f, indent=2)
        threshold = best_t
    with open(os.path.join(out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
        for it, s in zip(items, scores):
            f.write(json.dumps({"id": it["id"], "pred": 1 if s >= threshold else 0, "score": s}) + "\n")


def run_judge_baseline(items: list[dict], split: str, base_url: str = "http://localhost:11434"):
    print(f"\n--- Running llm_judge on split: {split} ({len(items)} items) ---")
    out_dir = os.path.join("results", f"llm_judge_{split}")
    os.makedirs(out_dir, exist_ok=True)
    res = [query_judge(it["passage1"], it["passage2"], base_url=base_url) for it in items]
    fails = sum(1 for _, ok in res if not ok)
    print(f"LLM Judge parse failures: {fails} / {len(items)}")
    with open(os.path.join(out_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
        for it, (pred, _) in zip(items, res):
            f.write(json.dumps({"id": it["id"], "pred": pred}) + "\n")


def run_evaluations_on_dev(gold_path: str):
    eval_py = os.path.join("scripts", "evaluate.py")
    baselines = [
        (["--preds", "results/nli_only_dev/preds.jsonl"], "nli_only_dev"),
        (["--preds", "results/llm_judge_dev/preds.jsonl"], "llm_judge_dev"),
        (["--baseline", "always_conflict"], "always_conflict_dev"),
        (["--baseline", "always_compatible"], "always_compatible_dev"),
    ]
    for opt, name in baselines:
        for extra in [[], ["--max_pair_tokens", "120"]]:
            cmd = [sys.executable, eval_py, "--gold", gold_path, "--split", "dev", "--name", f"{name}{'_tok120' if extra else ''}"] + opt + extra
            print(f"\n>>> Running evaluate.py {' '.join(cmd[2:])}")
            subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Run baselines (Part A)")
    p.add_argument("--split", required=True, choices=["dev", "test", "identical"])
    p.add_argument("--gold", default=os.path.join("data", "gold.jsonl"))
    p.add_argument("--gold_identical", default=os.path.join("data", "gold_identical.jsonl"))
    p.add_argument("--threshold_file", default=os.path.join("results", "nli_only_dev", "threshold.json"))
    p.add_argument("--base_url", default="http://localhost:11434")
    args = p.parse_args()

    tf = args.gold_identical if args.split == "identical" else args.gold
    with open(tf, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    items = all_items if args.split == "identical" else [it for it in all_items if it.get("split") == args.split]

    run_nli_baseline(items, args.split, args.threshold_file)
    run_judge_baseline(items, args.split, base_url=args.base_url)
    if args.split == "dev":
        run_evaluations_on_dev(args.gold)


if __name__ == "__main__":
    main()
