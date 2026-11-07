import math
import numpy as np


def confusion(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    return tp, fp, fn, tn


def precision(labels=None, preds=None, *, tp=None, fp=None):
    if tp is None: tp, fp, _, _ = confusion(labels, preds)
    return tp / (tp + fp) if (tp + fp) > 0 else None


def recall(labels=None, preds=None, *, tp=None, fn=None):
    if tp is None: tp, _, fn, _ = confusion(labels, preds)
    return tp / (tp + fn) if (tp + fn) > 0 else None


sensitivity = recall


def specificity(labels=None, preds=None, *, tn=None, fp=None):
    if tn is None: _, fp, _, tn = confusion(labels, preds)
    return tn / (tn + fp) if (tn + fp) > 0 else None


def f1(labels=None, preds=None, *, tp=None, fp=None, fn=None):
    if tp is None: tp, fp, fn, _ = confusion(labels, preds)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else None


f1_conflict = f1


def f1_compatible(labels=None, preds=None, *, tn=None, fp=None, fn=None):
    if tn is None: _, fp, fn, tn = confusion(labels, preds)
    denom = 2 * tn + fn + fp
    return (2 * tn) / denom if denom > 0 else None


def macro_f1(labels=None, preds=None, *, tp=None, fp=None, fn=None, tn=None):
    f1_pos = f1(labels, preds, tp=tp, fp=fp, fn=fn)
    f1_neg = f1_compatible(labels, preds, tn=tn, fp=fp, fn=fn)
    return (f1_pos + f1_neg) / 2.0 if (f1_pos is not None and f1_neg is not None) else None


def accuracy(labels=None, preds=None, *, tp=None, fp=None, fn=None, tn=None):
    if tp is None: tp, fp, fn, tn = confusion(labels, preds)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else None


def balanced_accuracy(labels=None, preds=None, *, tp=None, fp=None, fn=None, tn=None):
    rec = recall(labels, preds, tp=tp, fn=fn)
    spec = specificity(labels, preds, tn=tn, fp=fp)
    return (rec + spec) / 2.0 if (rec is not None and spec is not None) else None


def mcc(labels=None, preds=None, *, tp=None, fp=None, fn=None, tn=None):
    if tp is None: tp, fp, fn, tn = confusion(labels, preds)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def auroc(labels, scores):
    n1, n0 = sum(1 for l in labels if l == 1), sum(1 for l in labels if l == 0)
    if n1 == 0 or n0 == 0: return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]: j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j): ranks[order[k]] = avg_rank
        i = j
    sum_r1 = sum(ranks[idx] for idx, l in enumerate(labels) if l == 1)
    return (sum_r1 - (n1 * (n1 + 1)) / 2.0) / (n1 * n0)


def _call_metric(fn, items):
    try:
        return fn(items)
    except TypeError:
        return fn([it["label"] for it in items], [it["pred"] for it in items])


def cluster_bootstrap(items, metric_fn, n=2000, seed=42):
    articles = sorted(list({it["article"] for it in items}))
    if not articles: return (None, None), n
    by_art = {art: [it for it in items if it["article"] == art] for art in articles}
    art_list, num_arts = [by_art[art] for art in articles], len(articles)
    rng, vals, skipped = np.random.default_rng(seed), [], 0

    for _ in range(n):
        sample = [it for idx in rng.integers(0, num_arts, size=num_arts) for it in art_list[idx]]
        if not (any(it["label"] == 1 for it in sample) and any(it["label"] == 0 for it in sample)):
            skipped += 1
            continue
        v = _call_metric(metric_fn, sample)
        if v is None:
            skipped += 1
            continue
        vals.append(v)

    if not vals: return (None, None), skipped
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), skipped


def paired_bootstrap_diff(items, metric_fn_a, metric_fn_b, n=2000, seed=42):
    va, vb = _call_metric(metric_fn_a, items), _call_metric(metric_fn_b, items)
    diff = (va - vb) if (va is not None and vb is not None) else None
    articles = sorted(list({it["article"] for it in items}))
    if not articles: return diff, (None, None), n
    by_art = {art: [it for it in items if it["article"] == art] for art in articles}
    art_list, num_arts = [by_art[art] for art in articles], len(articles)
    rng, diffs, skipped = np.random.default_rng(seed), [], 0

    for _ in range(n):
        sample = [it for idx in rng.integers(0, num_arts, size=num_arts) for it in art_list[idx]]
        if not (any(it["label"] == 1 for it in sample) and any(it["label"] == 0 for it in sample)):
            skipped += 1
            continue
        ma, mb = _call_metric(metric_fn_a, sample), _call_metric(metric_fn_b, sample)
        if ma is None or mb is None:
            skipped += 1
            continue
        diffs.append(ma - mb)

    if not diffs: return diff, (None, None), skipped
    return diff, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))), skipped


def wilson_interval(k, n, z=1.96):
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n (got k={k}, n={n})")
    if n == 0:
        return (None, None)
    p = k / n
    z2 = z ** 2
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * (n ** 2)))) / denom
    lower = max(0.0, min(1.0, float(center - half)))
    upper = max(0.0, min(1.0, float(center + half)))
    return (lower, upper)

