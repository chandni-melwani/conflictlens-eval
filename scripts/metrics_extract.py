import statistics
from typing import Any, Iterable, List, Optional, Tuple, Union


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return " ".join(text.lower().split())


def _extract_record_info(rec: Any) -> Tuple[bool, List[dict], Optional[float], bool, str]:
    """
    Extracts (parse_ok, facts, latency, cached, passage) from various record representations.
    """
    if isinstance(rec, dict):
        if "res" in rec and isinstance(rec["res"], dict):
            res = rec["res"]
            parse_ok = bool(res.get("parse_ok", False))
            facts = res.get("facts", []) or []
            latency = res.get("latency")
            cached = bool(res.get("cached", False))
            passage = rec.get("passage", "")
        else:
            parse_ok = bool(rec.get("parse_ok", False))
            facts = rec.get("facts", []) or []
            latency = rec.get("latency")
            cached = bool(rec.get("cached", False))
            passage = rec.get("passage", "")
    else:
        parse_ok = getattr(rec, "parse_ok", False)
        facts = getattr(rec, "facts", []) or []
        latency = getattr(rec, "latency", None)
        cached = getattr(rec, "cached", False)
        passage = getattr(rec, "passage", "")

    return parse_ok, facts, latency, cached, passage


def parse_fail_rate(records: Iterable[Any]) -> float:
    """
    Share of passages with parse_ok false. Denominator is all passages.
    """
    recs = list(records)
    if not recs:
        return 0.0
    fails = sum(1 for r in recs if not _extract_record_info(r)[0])
    return fails / len(recs)


def zero_fact_rate(records: Iterable[Any]) -> float:
    """
    Share of passages with parse_ok true and an empty fact list.
    The denominator is all passages.
    """
    recs = list(records)
    if not recs:
        return 0.0
    zero_facts = sum(
        1 for r in recs
        if _extract_record_info(r)[0] and len(_extract_record_info(r)[1]) == 0
    )
    return zero_facts / len(recs)


def verbatim_value_rate(
    records_or_passage: Union[str, Iterable[Any]],
    values: Optional[Iterable[str]] = None,
) -> float:
    """
    Share of extracted values that occur as a substring of their source passage
    after lowercasing and collapsing whitespace.

    Can be called as:
      - verbatim_value_rate(passage_str, values_list)
      - verbatim_value_rate(records_list)
    """
    # Direct signature: (passage, values)
    if isinstance(records_or_passage, str) and values is not None:
        norm_passage = _normalize_text(records_or_passage)
        val_list = list(values)
        if not val_list:
            return 0.0
        matches = sum(1 for v in val_list if _normalize_text(v) in norm_passage)
        return matches / len(val_list)

    total_values = 0
    verbatim_matches = 0

    recs = list(records_or_passage) if not isinstance(records_or_passage, str) else []
    for item in recs:
        if isinstance(item, tuple) and len(item) == 2:
            p, vals = item
            norm_p = _normalize_text(p)
            for v in vals:
                total_values += 1
                if _normalize_text(v) in norm_p:
                    verbatim_matches += 1
        elif isinstance(item, dict):
            if "passage" in item and "values" in item:
                norm_p = _normalize_text(item["passage"])
                for v in item["values"]:
                    total_values += 1
                    if _normalize_text(v) in norm_p:
                        verbatim_matches += 1
            else:
                _, facts, _, _, passage = _extract_record_info(item)
                norm_p = _normalize_text(passage)
                for f in facts:
                    if isinstance(f, dict) and "value" in f:
                        total_values += 1
                        if _normalize_text(f["value"]) in norm_p:
                            verbatim_matches += 1
        else:
            _, facts, _, _, passage = _extract_record_info(item)
            norm_p = _normalize_text(passage)
            for f in facts:
                val = f.get("value") if isinstance(f, dict) else getattr(f, "value", None)
                if val is not None:
                    total_values += 1
                    if _normalize_text(val) in norm_p:
                        verbatim_matches += 1

    return verbatim_matches / total_values if total_values > 0 else 0.0


def numeric_entity_rate(entities_or_records: Iterable[Any]) -> float:
    """
    Share of extracted entities that contain no alphabetic character.

    Can be called as:
      - numeric_entity_rate(["1528", "Marlow"])
      - numeric_entity_rate(records)
    """
    items = list(entities_or_records)
    entities = []

    for item in items:
        if isinstance(item, str):
            entities.append(item)
        elif isinstance(item, dict):
            if "entity" in item and isinstance(item["entity"], str):
                entities.append(item["entity"])
            else:
                _, facts, _, _, _ = _extract_record_info(item)
                for f in facts:
                    if isinstance(f, dict) and "entity" in f:
                        entities.append(f["entity"])
        else:
            _, facts, _, _, _ = _extract_record_info(item)
            for f in facts:
                ent = f.get("entity") if isinstance(f, dict) else getattr(f, "entity", None)
                if ent is not None:
                    entities.append(str(ent))

    if not entities:
        return 0.0

    numeric_count = sum(1 for e in entities if not any(c.isalpha() for c in str(e)))
    return numeric_count / len(entities)


def median_facts(records: Iterable[Any]) -> float:
    """
    Median number of facts per passage.
    """
    recs = list(records)
    if not recs:
        return 0.0
    counts = [len(_extract_record_info(r)[1]) for r in recs]
    return float(statistics.median(counts))


def median_latency(records: Iterable[Any], fallback_to_cached: bool = True) -> Optional[float]:
    """
    Median latency of non-cached calls.
    If all calls are cached and fallback_to_cached is True, returns median of recorded latencies.
    """
    recs = list(records)
    if not recs:
        return 0.0
    non_cached = [
        _extract_record_info(r)[2]
        for r in recs
        if not _extract_record_info(r)[3] and _extract_record_info(r)[2] is not None
    ]
    if non_cached:
        return float(statistics.median(non_cached))
    if fallback_to_cached:
        stored = [
            _extract_record_info(r)[2]
            for r in recs
            if _extract_record_info(r)[2] is not None
        ]
        if stored:
            return float(statistics.median(stored))
    return None


def compute_extraction_metrics(records: Iterable[Any]) -> dict:
    """
    Computes all 6 automated metrics for a set of extraction records.
    """
    recs = list(records)
    return {
        "parse_fail_rate": parse_fail_rate(recs),
        "zero_fact_rate": zero_fact_rate(recs),
        "verbatim_value_rate": verbatim_value_rate(recs),
        "numeric_entity_rate": numeric_entity_rate(recs),
        "median_facts": median_facts(recs),
        "median_latency": median_latency(recs),
    }
