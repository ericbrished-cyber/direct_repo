import json
from typing import Any, Dict, List, Optional, Tuple

# ----------------- helpers ----------------- #

def normalize_name(s: Any) -> Optional[str]:
    if s is None:
        return None
    return " ".join(str(s).lower().split())


def _parse_rate(raw: Any) -> Optional[float]:
    """
    Accepts:
      - '60.0%'  -> 0.60
      - '60'     -> 0.60
      - '0.60'   -> 0.60
    Returns a float in [0,1] or None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()

    try:
        val = float(s)
    except ValueError:
        return None

    if is_percent:
        return val / 100.0

    # Heuristic if no '%' given
    if 0.0 <= val <= 1.0:
        return val        # already proportion
    if 1.0 < val <= 100.0:
        return val / 100.0  # treat as percent

    return None


def _parse_group_size(raw: Any) -> Optional[int]:
    """Parse group_size as positive integer."""
    if raw is None:
        return None
    s = str(raw).strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if v <= 0:
        return None
    return int(round(v))


def _ico_key(extraction_class: str, attrs: Dict[str, Any]) -> Tuple:
    """
    Build a key that identifies a specific ICO context.

    Includes:
      - normalized Intervention / Comparator
      - Outcome
      - Timepoint
      - Population
    """
    attrs = attrs or {}

    I = normalize_name(attrs.get("Intervention"))
    C = normalize_name(attrs.get("Comparator"))
    O = normalize_name(attrs.get("Outcome"))
    T = normalize_name(attrs.get("Timepoint"))
    P = normalize_name(attrs.get("Population"))

    # You could add more (e.g. Unit) if needed
    return (I, C, O, T, P)


RATE_TO_GROUPSIZE = {
    "intervention_rate": "intervention_group_size",
    "comparator_rate":   "comparator_group_size",
    "total_rate":        "total_group_size",
}

RATE_TO_EVENTS = {
    "intervention_rate": "intervention_events",
    "comparator_rate":   "comparator_events",
    "total_rate":        "total_events",
}

def _infer_events_for_document(extractions: list[dict]) -> list[dict]:
    """
    Given one document's extractions, return a new list with inferred *_events
    where *_rate + *_group_size exist for the same ICO.
    If there are no *_rate extractions, return the input unchanged.
    """
    # ---- STEP 1: quick check – any rates at all? ----
    has_rate = any(
        e.get("extraction_class") in RATE_TO_GROUPSIZE
        for e in extractions
    )
    if not has_rate:
        # nothing to do for this document
        return extractions

    # ---- rest of the logic as before ----
    new_extractions = list(extractions)

    group_size_by_key: dict[tuple[str, tuple], dict] = {}
    events_keys = set()

    for ex in extractions:
        cls = ex.get("extraction_class")
        attrs = ex.get("attributes") or {}
        key = _ico_key(cls, attrs)

        if cls in {"intervention_group_size", "comparator_group_size", "total_group_size"}:
            group_size_by_key[(cls, key)] = ex

        if cls in {"intervention_events", "comparator_events", "total_events"}:
            events_keys.add((cls, key))

    for rate_ex in extractions:
        rate_cls = rate_ex.get("extraction_class")
        if rate_cls not in RATE_TO_GROUPSIZE:
            continue

        attrs = rate_ex.get("attributes") or {}
        key = _ico_key(rate_cls, attrs)

        raw_rate = rate_ex.get("extraction_text")
        if raw_rate is None and "value" in rate_ex:
            raw_rate = rate_ex["value"]
        rate = _parse_rate(raw_rate)
        if rate is None:
            continue

        group_cls = RATE_TO_GROUPSIZE[rate_cls]
        events_cls = RATE_TO_EVENTS[rate_cls]

        if (events_cls, key) in events_keys:
            continue

        group_ex = group_size_by_key.get((group_cls, key))
        if group_ex is None:
            continue

        raw_n = group_ex.get("extraction_text")
        if raw_n is None and "value" in group_ex:
            raw_n = group_ex["value"]
        n = _parse_group_size(raw_n)
        if n is None:
            continue

        events = round(rate * n)
        events = max(0, min(events, n))

        new_extractions.append(
            {
                "extraction_class": events_cls,
                "extraction_text": str(int(events)),
                "attributes": attrs,
            }
        )
        events_keys.add((events_cls, key))

    return new_extractions


# ------------- public API ------------- #

def get_events_from_rate(jsonl_path: str, out_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Read a LangExtract JSONL file and infer *_events from *_rate + *_group_size
    for the SAME ICO (Intervention/Comparator + Outcome + Timepoint + Population).

    Returns:
        list of modified document dicts.
    If out_path is given, also writes a new JSONL with augmented extractions.
    """
    docs: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            exs = doc.get("extractions", [])
            doc["extractions"] = _infer_events_for_document(exs)
            docs.append(doc)

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

    return docs
