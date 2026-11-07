# ConflictLens Evaluation Package

## Purpose

This package provides reproduction and evaluation resources for the ConflictLens paper.
It contains saved model prediction records, ground truth evaluation labels,
evaluation and analysis scripts, unit tests, and paper report summaries.

## What is Included

- labels.csv: Split and binary contradiction labels (pair_id, split, label, article) for
  227 pairs across dev (51 items: 26 positive, 25 negative), test (151 items:
  78 positive, 73 negative), and identical text pairs (25 items: 25 positive).
  article is the Wikipedia article title, used for article-level bootstrap clustering.
- results/: Saved prediction records (preds.jsonl) and threshold/weight configurations
  for all evaluated systems across dev, test, and identical sets (ConflictLens
  hybrid exact, structured exact, structured fuzzy, structured none, NLI-only,
  LLM judge, and style-only).
- scripts/: Evaluation code, statistical metric functions, bootstrap confidence
  interval routines, and analysis utilities, including scripts/evaluate.py and
  scripts/metrics.py.
- tests/: Automated unit test suite verifying confusion matrix calculations,
  Wilson score intervals, and bootstrap metrics.
- splits/: The canonical article and pair partition definition (splits/split_v1.json).
- reports/: Summary tables, count reconciliation, cost tables, inter-rater and
  relabeling agreement, paired statistical tests, style checks, and metrics reports.

## What is NOT Included

- No Wikipedia passage text
- No extracted facts
- No raw data
- No model caches

## Labels

Positive pairs come from WikiContradict. Compatible (negative) pairs were selected and labelled by one annotator from the authors. 98 compatible pairs are used (25 dev, 73 test). A second co-author checked a sample of 45 items (see reports/coauthor_agreement.txt). The passages themselves are not included.

## Source and Attribution

- Positive contradiction pairs come from WikiContradict (Hugging Face
  ibm/Wikipedia_contradict_benchmark, commit c20e361f985ed480a659b35d98b49f2311fcd174,
  MIT license).
- Wikipedia text keeps its own attribution terms (Creative Commons
  Attribution-ShareAlike 4.0 International / GFDL).

## Recomputing Metrics

To recompute evaluation metrics from preds.jsonl plus labels.csv, use the
evaluator script scripts/evaluate.py.

Command format:
python scripts/evaluate.py --gold labels.csv --preds results/<folder>/preds.jsonl --split <split> --name <run_name>

Example (evaluating ConflictLens Hybrid Exact on the dev split):
python scripts/evaluate.py --gold labels.csv --preds results/conflictlens_hybrid_exact_dev/preds.jsonl --split dev --name conflictlens_hybrid_exact_dev

Example (evaluating ConflictLens Hybrid Exact on the test split):
python scripts/evaluate.py --gold labels.csv --preds results/conflictlens_hybrid_exact_test/preds.jsonl --split test --name conflictlens_hybrid_exact_test

The script outputs the confusion matrix (TP, FP, FN, TN), Precision, Recall,
Specificity, F1 (conflict), F1 (compatible), Macro-F1, Accuracy, Balanced
Accuracy, MCC, AUROC (when prediction scores exist), 95% Wilson intervals for
rates, and 95% bootstrap confidence intervals.

## License

Code, labels and predictions: MIT (see LICENSE). WikiContradict is MIT licensed by its authors. Wikipedia text keeps its own attribution terms and is not redistributed here.

## Tests

Run a test file with: python -m unittest tests\<file>.py
Eleven of the fifteen test files run with the standard library and the listed requirements. test_baselines.py, test_conflictlens.py and test_nli.py need torch and transformers installed. test_style_scores.py needs data\gold.jsonl, which is not distributed because it contains Wikipedia text. These four are included for reference.

## Contact

see the paper
