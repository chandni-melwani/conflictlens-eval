import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.cl_normalize import norm_text, numbers, parse_date, value_type
from scripts.nli import contradiction_score


def jaccard_sim(s1: str, s2: str) -> float:
    t1 = set(norm_text(s1).split())
    t2 = set(norm_text(s2).split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_candidate_pairs(facts1: list[dict], facts2: list[dict], blocking: str) -> tuple[list[tuple[dict, dict]], int]:
    pairs = []
    if blocking == "exact":
        k1, k2 = {}, {}
        for f in facts1:
            k1.setdefault((norm_text(f.get("entity", "")), norm_text(f.get("attribute", ""))), []).append(f)
        for f in facts2:
            k2.setdefault((norm_text(f.get("entity", "")), norm_text(f.get("attribute", ""))), []).append(f)
        common_keys = set(k1.keys()) & set(k2.keys())
        for key in common_keys:
            for f1 in k1[key]:
                for f2 in k2[key]:
                    pairs.append((f1, f2))
        return pairs, len(common_keys)
    elif blocking == "fuzzy":
        blocks = 0
        for f1 in facts1:
            for f2 in facts2:
                if jaccard_sim(f1.get("entity", ""), f2.get("entity", "")) >= 0.5 and \
                   jaccard_sim(f1.get("attribute", ""), f2.get("attribute", "")) >= 0.5:
                    pairs.append((f1, f2))
                    blocks += 1
        return pairs, blocks
    elif blocking == "none":
        for f1 in facts1:
            for f2 in facts2:
                pairs.append((f1, f2))
        return pairs, (1 if (facts1 and facts2) else 0)
    raise ValueError(f"Unknown blocking: {blocking}")


def check_pair(f1: dict, f2: dict, tau_sem: float = 0.5,
               enable_numeric: bool = True, enable_temporal: bool = True, enable_semantic: bool = True,
               nli_fn=contradiction_score) -> dict:
    vt1 = value_type(f1.get("attribute", ""), f1.get("value", ""))
    vt2 = value_type(f2.get("attribute", ""), f2.get("value", ""))

    if vt1 == "number" and vt2 == "number":
        detector = "numerical"
        if not enable_numeric:
            return {"detector": detector, "conflict": False, "score": 0.0}
        s1, s2 = numbers(f1.get("value", "")), numbers(f2.get("value", ""))
        conflict = bool(s1 and s2 and s1.isdisjoint(s2))
        return {"detector": detector, "conflict": conflict, "score": 1.0 if conflict else 0.0}
    elif vt1 == "date" and vt2 == "date":
        detector = "temporal"
        if not enable_temporal:
            return {"detector": detector, "conflict": False, "score": 0.0}
        d1, d2 = parse_date(f1.get("value", "")), parse_date(f2.get("value", ""))
        conflict = False
        if d1 and d2:
            y1, m1, day1 = d1
            y2, m2, day2 = d2
            if y1 is not None and y2 is not None and y1 != y2:
                conflict = True
            elif m1 is not None and m2 is not None and m1 != m2:
                conflict = True
            elif day1 is not None and day2 is not None and day1 != day2:
                conflict = True
        return {"detector": detector, "conflict": conflict, "score": 1.0 if conflict else 0.0}
    else:
        detector = "semantic"
        if not enable_semantic:
            return {"detector": detector, "conflict": False, "score": 0.0}
        prem = f"{f1.get('entity', '')} {f1.get('attribute', '')}: {f1.get('value', '')}"
        hypo = f"{f2.get('entity', '')} {f2.get('attribute', '')}: {f2.get('value', '')}"
        score = nli_fn(prem, hypo)
        return {"detector": detector, "conflict": (score >= tau_sem), "score": score}


def detect_conflicts(facts1: list[dict], facts2: list[dict], blocking: str = "exact", tau_sem: float = 0.5,
                     enable_numeric: bool = True, enable_temporal: bool = True, enable_semantic: bool = True,
                     nli_fn=contradiction_score) -> dict:
    pairs, n_blocks = get_candidate_pairs(facts1, facts2, blocking)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {
            "pred": 0, "score": 0.0, "no_pairs": True, "n_facts_p1": len(facts1),
            "n_facts_p2": len(facts2), "n_blocks": n_blocks, "n_candidate_pairs": 0,
            "flagged_detectors": [], "detector_types": [], "candidate_pairs": [],
        }

    evaluated = []
    for f1, f2 in pairs:
        res = check_pair(f1, f2, tau_sem=tau_sem, enable_numeric=enable_numeric,
                         enable_temporal=enable_temporal, enable_semantic=enable_semantic, nli_fn=nli_fn)
        evaluated.append({**res, "f1": f1, "f2": f2})

    flagged = [p for p in evaluated if p["conflict"]]
    max_score = max(p["score"] for p in evaluated)
    detector_types = [p["detector"] for p in flagged]

    return {
        "pred": 1 if flagged else 0,
        "score": max_score,
        "no_pairs": False,
        "n_facts_p1": len(facts1),
        "n_facts_p2": len(facts2),
        "n_blocks": n_blocks,
        "n_candidate_pairs": n_pairs,
        "flagged_detectors": detector_types,
        "detector_types": detector_types,
        "candidate_pairs": evaluated,
    }

