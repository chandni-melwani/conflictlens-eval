import re
import string

PUNCT_TABLE = str.maketrans("", "", string.punctuation)
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
DATE_KEYWORDS = {"year", "date", "born", "birth", "death", "died", "founded", "opened", "month"}
DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|"
    r"\b\d{1,2}\s+[a-zA-Z]+\s+\d{4}\b|"
    r"\b[a-zA-Z]+\s+\d{1,2},?\s+\d{4}\b|"
    r"\b[a-zA-Z]+\s+\d{4}\b"
)
NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![.\d])")


def norm_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    cleaned = s.lower().translate(PUNCT_TABLE)
    return " ".join(w for w in cleaned.split() if w != "the")


def parse_date(value: str) -> tuple[int, int | None, int | None] | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip().rstrip(".").strip()

    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", v)
    if m:
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (y, mth, d) if (1000 <= y <= 2100 and 1 <= mth <= 12 and 1 <= d <= 31) else None

    m = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", v)
    if m:
        d, mth_name, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return (y, MONTHS[mth_name], d) if (mth_name in MONTHS and 1000 <= y <= 2100 and 1 <= d <= 31) else None

    m = re.match(r"^([a-zA-Z]+)\s+(\d{1,2}),?\s+(\d{4})$", v)
    if m:
        mth_name, d, y = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        return (y, MONTHS[mth_name], d) if (mth_name in MONTHS and 1000 <= y <= 2100 and 1 <= d <= 31) else None

    m = re.match(r"^([a-zA-Z]+)\s+(\d{4})$", v)
    if m:
        mth_name, y = m.group(1).lower(), int(m.group(2))
        return (y, MONTHS[mth_name], None) if (mth_name in MONTHS and 1000 <= y <= 2100) else None

    m = re.match(r"^(\d{4})$", v)
    if m:
        y = int(m.group(1))
        return (y, None, None) if 1000 <= y <= 2100 else None

    return None


def numbers(value: str) -> set[float]:
    if not value or not isinstance(value, str):
        return set()
    val_no_dates = DATE_RE.sub(" ", value)
    res = set()
    for match in NUM_RE.finditer(val_no_dates):
        s = match.group(1).replace(",", "")
        try:
            res.add(float(s))
        except ValueError:
            pass
    return res


def value_type(attribute: str, value: str) -> str:
    norm_attr = norm_text(attribute)
    if parse_date(value) is not None and any(kw in norm_attr for kw in DATE_KEYWORDS):
        return "date"
    if len(numbers(value)) > 0:
        return "number"
    return "text"

