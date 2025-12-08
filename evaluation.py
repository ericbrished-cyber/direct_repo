import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from utils import get_pmcid_from_filename

# =======================
# Normalisation helpers
# =======================

def normalize_name(s: str):
    """
    Normalize arm / outcome / intervention names for matching.

    - Unicode normalize (NFKC)
    - Lowercase
    - Fix common microgram encodings: _g, µg, μg → ug
    - Collapse whitespace
    """
    if s is None:
        return None

    # 1) Unicode normalization (handles weird composed characters)
    s = unicodedata.normalize("NFKC", str(s))

    # 2) Lowercase
    s = s.lower()

    # 3) Fix common microgram variants:
    #    "300 _g", "300µg", "300 μg" → "300 ug"
    s = re.sub(r'(\d+)\s*[_µμ]\s*g\b', r'\1 ug', s)

    # Fallback: any stray "_g", "µg", "μg" → " ug"
    s = re.sub(r'[_µμ]\s*g\b', ' ug', s)

    # Same for "/kg" forms: "3 _g/kg", "3µg/kg" → "3 ug/kg"
    s = re.sub(r'(\d+)\s*[_µμ]\s*g/kg\b', r'\1 ug/kg', s)
    s = re.sub(r'[_µμ]\s*g/kg\b', ' ug/kg', s)

    # 4) Collapse whitespace
    s = " ".join(s.split())

    return s


def normalize_value(v):
    """
    Normalize numeric-ish values for comparison.

    - Treat common missing markers as None.
    - Strip '%' and parse as float if possible.
    """
    if v is None:
        return None

    s = str(v).strip().lower()
    if s in {"", "none", "nr", "not reported", "n/a", "na", "not extractable"}:
        return None

    # Remove percentage sign but keep the numeric scale
    if s.endswith("%"):
        s = s[:-1].strip()

    try:
        return float(s)
    except ValueError:
        return None





# =======================
# Schema mapping
# =======================

# Map gold/extraction classes to (role, field)
# role: "I" = intervention arm, "C" = comparator arm
# field: logical variable name used in evaluation
CLASS_INFO = {
    # counts
    "intervention_events": ("I", "events"),
    "comparator_events":   ("C", "events"),

    # group sizes
    "intervention_group_size": ("I", "group_size"),
    "intervention_groupsize":  ("I", "group_size"),
    "group_size_intervention": ("I", "group_size"),

    "comparator_group_size": ("C", "group_size"),
    "comparator_groupsize":  ("C", "group_size"),
    "group_size_comparator": ("C", "group_size"),

    # means
    "intervention_mean": ("I", "mean"),
    "mean_intervention": ("I", "mean"),

    "comparator_mean": ("C", "mean"),
    "mean_comparator": ("C", "mean"),

    # standard deviations
    "intervention_standard_deviation": ("I", "sd"),
    "sd_intervention":                 ("I", "sd"),

    "comparator_standard_deviation": ("C", "sd"),
    "sd_comparator":                 ("C", "sd"),

    # rates (percentage outcomes)
    "intervention_rate": ("I", "rate"),
    "comparator_rate":   ("C", "rate"),
}

# Direct-field output mapping (no extraction_class key)
DIRECT_FIELD_INFO = {
    "intervention_group_size": ("I", "group_size"),
    "comparator_group_size": ("C", "group_size"),
    "intervention_events": ("I", "events"),
    "comparator_events": ("C", "events"),
    "intervention_rate": ("I", "rate"),
    "comparator_rate": ("C", "rate"),
    "intervention_mean": ("I", "mean"),
    "comparator_mean": ("C", "mean"),
    "intervention_standard_deviation": ("I", "sd"),
    "comparator_standard_deviation": ("C", "sd"),
}


# ==================================
# Gold side: build arm-level facts
# ==================================

def build_gold_arm_facts(gold_path: str, pmcid: int):
    """
    Build a set of arm-level facts from the gold-standard JSON.

    Each fact is a tuple:
      (pmcid, outcome_norm, role, arm_name_norm, field, value)

    where:
      - pmcid: int
      - outcome_norm: normalized outcome name
      - role: "I" or "C"
      - arm_name_norm: normalized intervention/comparator name
      - field: logical field name ("events", "group_size", "mean", "sd", "rate")
      - value: normalized float value
    """
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_rows = json.load(f)

    facts = set()

    for row in gold_rows:
        if int(row["pmcid"]) != int(pmcid):
            continue

        outcome = normalize_name(row.get("outcome"))
        I_name = normalize_name(row.get("intervention"))
        C_name = normalize_name(row.get("comparator"))

        # For each possible class, see if this row has a value
        for cls, (role, field) in CLASS_INFO.items():
            raw_val = row.get(cls)
            val = normalize_value(raw_val)
            if val is None:
                continue

            # Pick the right arm for this role
            if role == "I":
                arm = I_name
            else:
                arm = C_name

            if arm is None:
                continue  # skip rows with missing arm name (defensive)

            fact = (pmcid, outcome, role, arm, field, val)
            facts.add(fact)

    return facts


# ==================================
# Prediction side: build arm-facts
# ==================================

def build_pred_arm_facts(jsonl_path: str):
    """
    Build a set of arm-level facts from model outputs (direct JSON or legacy LangExtract JSONL).

    Each fact is in the same form as gold:
      (pmcid, outcome_norm, role, arm_name_norm, field, value)
    """
    pmcid = get_pmcid_from_filename(jsonl_path)
    facts = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            ex_list = doc.get("extractions", [])
            if not ex_list:
                continue

            # Detect direct-field schema (no extraction_class)
            if all("extraction_class" not in ex for ex in ex_list):
                facts |= _build_facts_from_direct_fields(pmcid, ex_list)
                continue

            for e in ex_list:
                cls = e.get("extraction_class")
                info = CLASS_INFO.get(cls)
                if info is None:
                    # ignore extraction classes we don't evaluate
                    continue

                role, field = info
                attrs = e.get("attributes") or {}
                outcome = normalize_name(attrs.get("Outcome"))

                if role == "I":
                    arm = normalize_name(attrs.get("Intervention"))
                else:
                    arm = normalize_name(attrs.get("Comparator"))

                if outcome is None or arm is None:
                    # can't place this extraction on an arm + outcome
                    continue

                val = normalize_value(e.get("extraction_text"))
                if val is None:
                    continue

                fact = (pmcid, outcome, role, arm, field, val)
                facts.add(fact)

    return facts


def _build_facts_from_direct_fields(
    pmcid: int,
    rows: Iterable[Dict[str, Any]],
):
    """
    Convert the new JSON shape (one dict per ICO row) into evaluation facts.
    """
    facts = set()
    for row in rows:
        outcome = normalize_name(row.get("outcome"))
        I_name = normalize_name(row.get("intervention"))
        C_name = normalize_name(row.get("comparator"))

        if outcome is None or (I_name is None and C_name is None):
            continue

        for field_name, (role, logical_field) in DIRECT_FIELD_INFO.items():
            val = normalize_value(row.get(field_name))
            if val is None:
                continue
            arm = I_name if role == "I" else C_name
            if arm is None:
                continue
            facts.add((pmcid, outcome, role, arm, logical_field, val))
    return facts


# ==================================
# Open-world evaluation logic
# ==================================

def _dict_from_facts(facts):
    """
    Convert a set of facts:
      (pmcid, outcome, role, arm, field, value)
    into a dict: key -> value, where
      key = (pmcid, outcome, role, arm, field).
    """
    d: Dict[Tuple[Any, ...], Any] = {}
    for pmcid, outcome, role, arm, field, value in facts:
        key = (pmcid, outcome, role, arm, field)
        d[key] = value
    return d


def evaluate_arm_facts_open_world(gold_facts, pred_facts):
    """
    Open-world evaluation:

    - Only cells (pmcid, outcome, role, arm, field) that exist in the gold
      are used for TP / FP / FN.
    - Predictions for cells not in the gold at all are treated as
      'extra_predictions', not as false positives.

    Returns a dict with:
      - tp: number of correctly predicted gold cells
      - fp_in_gold: number of gold cells that were predicted with the wrong value
      - fn: number of gold cells that were not predicted correctly
      - extra_predictions: number of predicted cells that don't exist in gold
      - precision, recall, f1: over gold-defined cells only
      - plus key-level breakdowns for inspection
    """
    gold_dict = _dict_from_facts(gold_facts)
    pred_dict = _dict_from_facts(pred_facts)

    gold_keys = set(gold_dict.keys())
    pred_keys = set(pred_dict.keys())

    shared_keys = gold_keys & pred_keys
    extra_keys = pred_keys - gold_keys         # predicted cells outside gold
    missing_keys = gold_keys - pred_keys       # gold cells with no prediction

    # Figure out which shared keys are exact matches and which are mismatches
    tp_keys = set()
    fp_keys = set()          # wrong value on a gold cell
    mismatched_keys = set()  # same as fp_keys, for clarity

    for k in shared_keys:
        g_val = gold_dict[k]
        p_val = pred_dict[k]
        if g_val == p_val:
            tp_keys.add(k)
        else:
            fp_keys.add(k)
            mismatched_keys.add(k)

    # FN: gold cell either missing or mismatched
    fn_keys = missing_keys | mismatched_keys

    tp = len(tp_keys)
    fp_in_gold = len(fp_keys)
    fn = len(fn_keys)
    extra = len(extra_keys)

    precision = tp / (tp + fp_in_gold) if (tp + fp_in_gold) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp_in_gold": fp_in_gold,
        "fn": fn,
        "extra_predictions": extra,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_keys": tp_keys,
        "fp_keys": fp_keys,
        "fn_keys": fn_keys,
        "extra_keys": extra_keys,
        "gold_dict": gold_dict,
        "pred_dict": pred_dict,
    }
