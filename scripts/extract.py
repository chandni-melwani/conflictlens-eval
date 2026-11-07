import hashlib
import json
import os
import time
from datetime import datetime, timezone
import requests

MODEL = "qwen2.5:3b"
PROMPT_VERSION = "extract_v1"
CACHE_DIR = os.path.join("data", "cache", "extractions")

EXTRACTION_PROMPT = """Extract every factual claim in the passage as (entity, attribute, value) triples.
Copy the wording from the passage. Keep numbers and dates exactly as written.
Do not add facts, do not infer, do not judge whether anything conflicts.
If the passage has no claims, return an empty list.
Output only JSON of the form {"facts": [{"entity": "...", "attribute": "...", "value": "..."}]}.

Example (do not use in output):
Passage: The Celestia Sky Bridge was opened in 2042 and measures 1,250 meters in length.
Output: {"facts": [{"entity": "Celestia Sky Bridge", "attribute": "opened in", "value": "2042"}, {"entity": "Celestia Sky Bridge", "attribute": "length", "value": "1,250 meters"}]}

Passage:
{passage}
"""

PROMPT_V1 = EXTRACTION_PROMPT

PROMPT_V2 = """You read a passage and list its facts as (entity, attribute, value) triples.

Rules:
- entity: the person, place, event, organization or object the fact is about. An entity is never a bare date or number.
- attribute: the property being stated, for example year, month, length, number of survivors, winner, location, cause.
- value: the number, date, name or phrase exactly as written in the passage, including units. Never shorten it.
- List every fact that contains a number, date, name, place or event. If the passage contains any such content, return at least one fact.
- If the passage is a table or a list, return one fact per cell you can read, at most 12 facts.
- Copy wording from the passage. Do not add facts, do not infer, do not say whether anything conflicts.
- Output only JSON in this form: {"facts": [{"entity": "...", "attribute": "...", "value": "..."}]}

Examples (invented, not from any dataset):

Passage: The Zorin Bridge opened in 1932 and is 450 metres long.
Output: {"facts": [{"entity": "Zorin Bridge", "attribute": "opening year", "value": "1932"}, {"entity": "Zorin Bridge", "attribute": "length", "value": "450 metres"}]}

Passage: After the 1887 flood, the town of Marlow lost 40 of its 120 homes.
Output: {"facts": [{"entity": "Marlow", "attribute": "year of flood", "value": "1887"}, {"entity": "Marlow", "attribute": "homes lost", "value": "40"}, {"entity": "Marlow", "attribute": "total homes", "value": "120"}]}

Passage: Stage | Date | Winner 1 | 3 May | A. Roe (ENG)
Output: {"facts": [{"entity": "Stage 1", "attribute": "date", "value": "3 May"}, {"entity": "Stage 1", "attribute": "winner", "value": "A. Roe (ENG)"}]}

Now the passage:
{passage}
Output:
"""

EXTRACTION_PROMPT_V2 = PROMPT_V2

PROMPTS = {
    "extract_v1": PROMPT_V1,
    "extract_v2": PROMPT_V2,
}

STRICT_REMINDER = (
    "\n\nCRITICAL: Your output MUST be strictly valid JSON containing a 'facts' list "
    "where each item is an object with non-empty string 'entity', 'attribute', and 'value' fields. "
    "Format: {\"facts\": [{\"entity\": \"...\", \"attribute\": \"...\", \"value\": \"...\"}]}. Output nothing else."
)


def validate_facts(raw_text: str) -> tuple[bool, list[dict]]:
    if not isinstance(raw_text, str):
        return False, []
    try:
        data = json.loads(raw_text.strip())
    except Exception:
        return False, []
    if not isinstance(data, dict):
        return False, []
    facts = data.get("facts")
    if not isinstance(facts, list):
        return False, []
    validated = []
    for item in facts:
        if not isinstance(item, dict):
            return False, []
        entity = item.get("entity")
        attr = item.get("attribute")
        val = item.get("value")
        if not (isinstance(entity, str) and entity.strip() and
                isinstance(attr, str) and attr.strip() and
                isinstance(val, str) and val.strip()):
            return False, []
        validated.append({"entity": entity, "attribute": attr, "value": val})
    return True, validated


def get_cache_key(passage: str, model: str = MODEL, prompt_version: str = PROMPT_VERSION) -> str:
    key_src = f"{model}{prompt_version}{passage}"
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()


def get_cache_path(passage: str, cache_dir: str = CACHE_DIR, model: str = MODEL, prompt_version: str = PROMPT_VERSION) -> str:
    key = get_cache_key(passage, model=model, prompt_version=prompt_version)
    return os.path.join(cache_dir, f"{key}.json")


def _call_ollama(prompt: str, base_url: str = "http://localhost:11434") -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. Please start Ollama with '{MODEL}'."
        ) from e


def extract_facts(
    passage: str,
    base_url: str = "http://localhost:11434",
    cache_dir: str = CACHE_DIR,
    model: str = MODEL,
    prompt_version: str = PROMPT_VERSION,
    network_caller=None,
) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    c_path = get_cache_path(passage, cache_dir=cache_dir, model=model, prompt_version=prompt_version)
    if os.path.exists(c_path):
        with open(c_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        res["cached"] = True
        return res

    caller = network_caller if network_caller is not None else _call_ollama
    start_time = time.perf_counter()
    if prompt_version not in PROMPTS:
        raise ValueError(f"Unknown prompt_version: {prompt_version}. Available: {list(PROMPTS.keys())}")
    template = PROMPTS[prompt_version]
    prompt = template.replace("{passage}", passage)

    raw_output = caller(prompt)
    parse_ok, facts = validate_facts(raw_output)

    if not parse_ok:
        retry_prompt = prompt + STRICT_REMINDER
        raw_output = caller(retry_prompt)
        parse_ok, facts = validate_facts(raw_output)
        if not parse_ok:
            facts = []

    latency = time.perf_counter() - start_time
    now_ts = datetime.now(timezone.utc).isoformat()

    result = {
        "raw_output": raw_output,
        "facts": facts,
        "parse_ok": parse_ok,
        "latency": latency,
        "timestamp": now_ts,
        "model": model,
        "MODEL": model,
        "prompt_version": prompt_version,
        "PROMPT_VERSION": prompt_version,
    }

    with open(c_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result_copy = dict(result)
    result_copy["cached"] = False
    return result_copy
