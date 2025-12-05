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


def print_mismatches_open_world(gold_facts, pred_facts):
    """
    Pretty-print:
      - summary of counts
      - details of gold cells not correctly predicted
      - details of extra predictions with no gold cell
    """
    analysis = evaluate_arm_facts_open_world(gold_facts, pred_facts)

    gold_dict = analysis["gold_dict"]
    pred_dict = analysis["pred_dict"]

    tp_keys = analysis["tp_keys"]
    fp_keys = analysis["fp_keys"]
    fn_keys = analysis["fn_keys"]
    extra_keys = analysis["extra_keys"]

    print("\n" + "=" * 80)
    print("SUMMARY (open-world)")
    print("=" * 80)
    print(f"TP (on gold cells):              {analysis['tp']}")
    print(f"FP (wrong value on gold cells):  {analysis['fp_in_gold']}")
    print(f"FN (gold cells missed):          {analysis['fn']}")
    print(f"Extra predictions (no gold cell):{analysis['extra_predictions']}")
    print(f"Precision (gold cells):          {analysis['precision']:.3f}")
    print(f"Recall (gold cells):             {analysis['recall']:.3f}")
    print(f"F1:                              {analysis['f1']:.3f}")

    def fmt_key_val(key, val, label_role=None):
        pmcid, outcome, role, arm, field = key
        role_label = label_role or ("Intervention" if role == "I" else "Comparator")
        return (
            f"PMCID={pmcid}, Outcome={outcome!r}, "
            f"{role_label}={arm!r}, field={field!r}, value={val}"
        )

    # Gold cells that were not correctly predicted (either missing or mismatched)
    if fn_keys:
        print("\n" + "=" * 80)
        print("GOLD CELLS NOT CORRECTLY PREDICTED (FN)")
        print("=" * 80)
        for k in sorted(fn_keys):
            print("  GOLD:", fmt_key_val(k, gold_dict[k]))
            if k in pred_dict:
                print("  PRED:", fmt_key_val(k, pred_dict[k]))
            else:
                print("  PRED: <missing>")
            print("-" * 40)

    # Predictions on cells the gold doesn't have at all
    if extra_keys:
        print("\n" + "=" * 80)
        print("EXTRA PREDICTIONS (no corresponding gold cell)")
        print("=" * 80)
        for k in sorted(extra_keys):
            print("  PRED:", fmt_key_val(k, pred_dict[k], label_role="Arm"))


# ==================================
# Top-level convenience function
# ==================================

def evaluate_file(extraction_file: str, gold_file: str):
    """
    Full pipeline for a single paper:

    Inputs:
      - extraction_file: JSONL with LangExtract outputs for one PMCID
      - gold_file: JSON with gold-standard annotated rows (all PMCIDs)

    Steps:
      1. Infer PMCID from extraction filename.
      2. Build gold arm-facts for that PMCID.
      3. Build predicted arm-facts from model output.
      4. Run open-world evaluation (only gold-defined cells contribute to TP/FP/FN).
      5. Print summary + mismatches.
    """
    pmcid = get_pmcid_from_filename(extraction_file)
    gold_facts = build_gold_arm_facts(gold_file, pmcid)
    pred_facts = build_pred_arm_facts(extraction_file)

    analysis = evaluate_arm_facts_open_world(gold_facts, pred_facts)

    print(f"\nEvaluating PMCID: {pmcid}")
    print(f"Gold arm-facts: {len(gold_facts)}")
    print(f"Pred arm-facts: {len(pred_facts)}")

    print(f"TP (on gold cells):       {analysis['tp']}")
    print(f"FP (wrong on gold cells): {analysis['fp_in_gold']}")
    print(f"FN (gold cells missed):   {analysis['fn']}")
    print(f"Extra predictions:        {analysis['extra_predictions']}")
    print(f"Precision (gold cells):   {analysis['precision']:.3f}")
    print(f"Recall (gold cells):      {analysis['recall']:.3f}")
    print(f"F1:                       {analysis['f1']:.3f}")

    print_mismatches_open_world(gold_facts, pred_facts)


# ==================================
# CLI usage
# ==================================

if __name__ == "__main__":
    # Adjust these paths to your setup
    extraction_path = "outputs/gpt5_direct_pdf_guided/3687098_guided.jsonl"
    gold_path = "gold-standard/annotated_rct_dataset.json"
    evaluate_file(extraction_path, gold_path)
