import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.metrics import auroc

PRONOUN_RE = re.compile(r"^\s*(?:He|She|They|It|His|Her|Their|Its|This|These|Those|That)\b")


def starts_with_pronoun(text: str) -> bool:
    return bool(PRONOUN_RE.match(text or ""))


def compute_passage_stats(passages: list[str]) -> dict:
    n = len(passages)
    if n == 0:
        return {"pronoun": 0.0, "paren": 0.0, "semicolon": 0.0, "mean_words": 0.0, "period": 0.0}
    return {
        "pronoun": sum(1 for p in passages if starts_with_pronoun(p)) / n,
        "paren": sum(1 for p in passages if "(" in p) / n,
        "semicolon": sum(1 for p in passages if ";" in p) / n,
        "mean_words": sum(len(p.split()) for p in passages) / n,
        "period": sum(1 for p in passages if p.strip().endswith(".")) / n,
    }


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, iters: int = 15, l2: float = 1e-3):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_s = (X - mu) / sigma
    X_b = np.c_[np.ones(len(X)), X_s]
    w = np.zeros(X_b.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X_b @ w, -25, 25)))
        grad = X_b.T @ (p - y) / len(y) + (l2 * np.r_[0, w[1:]]) / len(y)
        W = np.diag(p * (1.0 - p))
        H = (X_b.T @ W @ X_b) / len(y) + (l2 * np.eye(len(w))) / len(y)
        w -= np.linalg.solve(H, grad)
    return w, mu, sigma


def predict_proba(X: np.ndarray, w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> list[float]:
    X_s = (X - mu) / sigma
    X_b = np.c_[np.ones(len(X)), X_s]
    return (1.0 / (1.0 + np.exp(-np.clip(X_b @ w, -25, 25)))).tolist()


def extract_features(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for it in items:
        p1 = it.get("passage1", "")
        p2 = it.get("passage2", "")
        X.append([
            float(it.get("pair_tokens", 0)),
            1.0 if starts_with_pronoun(p1) else 0.0,
            1.0 if starts_with_pronoun(p2) else 0.0,
            1.0 if it.get("both_digit", False) else 0.0,
        ])
        y.append(int(it["label"]))
    return np.array(X, dtype=float), np.array(y, dtype=float)


def save_style_test_predictions(
    gold_path: str = os.path.join("data", "gold.jsonl"),
    out_dir: str = os.path.join("results", "style_only_test"),
) -> dict:
    with open(gold_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    dev_items = [it for it in items if it.get("split") == "dev"]
    test_items = [it for it in items if it.get("split") == "test"]

    X_dev, y_dev = extract_features(dev_items)
    X_test, y_test = extract_features(test_items)

    w, mu, sigma = fit_logistic_regression(X_dev, y_dev)
    test_scores = predict_proba(X_test, w, mu, sigma)

    test_auroc = auroc([int(y) for y in y_test], test_scores)

    os.makedirs(out_dir, exist_ok=True)
    preds_file = os.path.join(out_dir, "preds.jsonl")
    with open(preds_file, "w", encoding="utf-8") as f:
        for it, score in zip(test_items, test_scores):
            row = {
                "id": it["id"],
                "label": int(it["label"]),
                "pred": 1 if score >= 0.5 else 0,
                "score": float(score),
            }
            f.write(json.dumps(row) + "\n")

    weights_file = os.path.join(out_dir, "weights.json")
    weights_data = {
        "features": ["pair_tokens", "p1_starts_pronoun", "p2_starts_pronoun", "both_digit"],
        "w": w.tolist(),
        "intercept": float(w[0]),
        "weights": w.tolist(),
        "coefficients": w[1:].tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }
    with open(weights_file, "w", encoding="utf-8") as f:
        json.dump(weights_data, f, indent=2)

    return {
        "test_auroc": test_auroc,
        "w": w,
        "mu": mu,
        "sigma": sigma,
        "preds_file": preds_file,
        "weights_file": weights_file,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Style checks and metadata-only classifier")
    parser.add_argument("--gold", default=os.path.join("data", "gold.jsonl"))
    parser.add_argument("--out_dir", default=os.path.join("results", "style_only_test"))
    parser.add_argument("--report", default=os.path.join("reports", "style_checks.txt"))
    args = parser.parse_args()

    gold_path = args.gold
    with open(gold_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    lines = ["=" * 70, "STYLE CHECKS AND SURFACE ARTIFACTS REPORT", "=" * 70, ""]
    for sp in ["dev", "test"]:
        for lbl, lbl_name in [(1, "Positive (Conflict)"), (0, "Negative (Compatible)")]:
            sub = [it for it in items if it.get("split") == sp and it.get("label") == lbl]
            passages = [p for it in sub for p in (it.get("passage1", ""), it.get("passage2", ""))]
            stats = compute_passage_stats(passages)
            lines.append(f"Split: {sp.upper()} | Class: {lbl} ({lbl_name}) | Pairs: {len(sub)} | Passages: {len(passages)}")
            lines.append(f"  Share starting with pronoun/demonstrative: {stats['pronoun']:.4f}")
            lines.append(f"  Share containing '(':                      {stats['paren']:.4f}")
            lines.append(f"  Share containing ';':                      {stats['semicolon']:.4f}")
            lines.append(f"  Mean words per passage:                    {stats['mean_words']:.2f}")
            lines.append(f"  Share ending with a period:                {stats['period']:.4f}")
            lines.append("")

    dev_items = [it for it in items if it.get("split") == "dev"]
    test_items = [it for it in items if it.get("split") == "test"]

    X_dev, y_dev = extract_features(dev_items)
    X_test, y_test = extract_features(test_items)

    w, mu, sigma = fit_logistic_regression(X_dev, y_dev)
    dev_scores = predict_proba(X_dev, w, mu, sigma)
    test_scores = predict_proba(X_test, w, mu, sigma)

    dev_auroc = auroc([int(y) for y in y_dev], dev_scores)
    test_auroc = auroc([int(y) for y in y_test], test_scores)

    # Save test predictions and dev weights
    save_style_test_predictions(gold_path=gold_path, out_dir=args.out_dir)

    lines.extend([
        "=" * 70,
        "METADATA-ONLY CLASSIFIER (LOGISTIC REGRESSION)",
        "=" * 70,
        "Features: pair_tokens, p1_starts_pronoun, p2_starts_pronoun, both_digit",
        "Trained on: dev split only",
        "NOTE: This classifier uses only metadata and uses no passage meaning.",
        "",
        f"Dev AUROC (training): {dev_auroc:.4f}" if dev_auroc is not None else "Dev AUROC: undefined",
        f"Test AUROC:           {test_auroc:.4f}" if test_auroc is not None else "Test AUROC: undefined",
        "=" * 70,
    ])

    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")
    print(report_content)


if __name__ == "__main__":
    main()

