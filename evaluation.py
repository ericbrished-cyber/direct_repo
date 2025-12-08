"""
Evaluation utilities.

Primary flow mirrors eval.py: fuzzy row matching on outcome/intervention/comparator,
numeric comparison with tolerance, and field-level TP/TN/FP/FN counts.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List

# Default gold path (single source of truth)
GOLD_PATH_DEFAULT = "gold-standard/gold_standard_clean.json"
# === Configuration (aligned with eval.py) ===
MATCH_THRESHOLD = 0.8
NUMBER_TOLERANCE = 0.01

# Numeric fields we score
NUMERIC_FIELDS = [
    "intervention_group_size",
    "comparator_group_size",
    "intervention_events",
    "comparator_events",
    "intervention_mean",
    "comparator_mean",
    "intervention_standard_deviation",
    "comparator_standard_deviation",
]


# === New evaluation logic (from eval.py) ===
def load_json_file(filepath: str) -> Any:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}")
        sys.exit(1)


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def calculate_similarity(pred_row: Dict[str, Any], gold_row: Dict[str, Any]) -> float:
    pred_str = (
        f"{normalize_text(pred_row.get('outcome'))} | "
        f"{normalize_text(pred_row.get('intervention'))} | "
        f"{normalize_text(pred_row.get('comparator'))}"
    )
    gold_str = (
        f"{normalize_text(gold_row.get('outcome'))} | "
        f"{normalize_text(gold_row.get('intervention'))} | "
        f"{normalize_text(gold_row.get('comparator'))}"
    )
    return difflib.SequenceMatcher(None, pred_str, gold_str).ratio()


def compare_numbers(pred_val: Any, gold_val: Any) -> bool:
    """Return True if numbers match within tolerance."""
    try:
        if isinstance(pred_val, str):
            pred_val = float(pred_val.replace(",", ""))
        if isinstance(gold_val, str):
            gold_val = float(gold_val.replace(",", ""))
        return math.isclose(float(pred_val), float(gold_val), rel_tol=NUMBER_TOLERANCE)
    except (ValueError, TypeError):
        return False


def _group_gold_by_pmcid(gold_rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for idx, entry in enumerate(gold_rows):
        pmcid = entry.get("pmcid")
        grouped.setdefault(pmcid, []).append({"data": entry, "index": idx})
    return grouped


def _normalize_pmcid(value: Any) -> Any:
    # Try to keep integers as integers; otherwise return as-is.
    try:
        return int(value)
    except Exception:
        return value


def _extract_rows(pred_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return pred_obj.get("rows") or pred_obj.get("extractions") or []


def _group_predictions(pred_data: Any) -> List[Dict[str, Any]]:
    """
    Normalize predictions into a list of {"pmcid": pmcid, "rows": [...]},
    accepting the current output schema (single dict with 'extractions') and
    the eval.py grouped/flat variants.
    """
    if isinstance(pred_data, dict):
        rows = _extract_rows(pred_data)
        pmcid = _normalize_pmcid(pred_data.get("pmcid") or (rows[0].get("pmcid") if rows else None))
        return [{"pmcid": pmcid, "rows": rows}]

    if isinstance(pred_data, list) and pred_data and "rows" not in pred_data[0] and "extractions" not in pred_data[0]:
        grouped: Dict[Any, Dict[str, Any]] = {}
        for row in pred_data:
            pid = _normalize_pmcid(row.get("pmcid"))
            grouped.setdefault(pid, {"pmcid": pid, "rows": []})
            grouped[pid]["rows"].append(row)
        return list(grouped.values())

    if isinstance(pred_data, list):
        grouped_objs = []
        for obj in pred_data:
            rows = _extract_rows(obj)
            pmcid = _normalize_pmcid(obj.get("pmcid") or (rows[0].get("pmcid") if rows else None))
            grouped_objs.append({"pmcid": pmcid, "rows": rows})
        return grouped_objs

    raise ValueError("Unsupported prediction format")


def evaluate(gold_file: str, pred_file: str, verbose: bool = True) -> Dict[str, Any]:
    gold_data_all = load_json_file(gold_file)
    pred_data = load_json_file(pred_file)

    grouped_predictions = _group_predictions(pred_data)
    pmcid_filter = {pred_obj.get("pmcid") for pred_obj in grouped_predictions}
    gold_data = [row for row in gold_data_all if row.get("pmcid") in pmcid_filter]

    gold_by_pmcid = _group_gold_by_pmcid(gold_data)
    all_matched_gold_indices = set()

    tp = tn = fp = fn = 0
    per_field_counts: Dict[str, Dict[str, int]] = {
        field: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for field in NUMERIC_FIELDS
    }

    if verbose:
        print(f"--- Evaluation ({gold_file} vs {pred_file}) ---")

    for pred_obj in grouped_predictions:
        pmcid = pred_obj.get("pmcid")
        rows = pred_obj.get("rows") or []

        if pmcid not in gold_by_pmcid:
            for row in rows:
                for field in NUMERIC_FIELDS:
                    if row.get(field) is not None:
                        fp += 1
                        per_field_counts[field]["fp"] += 1
            continue

        gold_candidates = gold_by_pmcid[pmcid]
        study_matched_indices = set()

        for pred_row in rows:
            best_score = -1.0
            best_gold_wrapper = None

            for wrapper in gold_candidates:
                if wrapper["index"] in study_matched_indices:
                    continue
                score = calculate_similarity(pred_row, wrapper["data"])
                if score > best_score:
                    best_score = score
                    best_gold_wrapper = wrapper

            if best_score >= MATCH_THRESHOLD and best_gold_wrapper:
                gold_row = best_gold_wrapper["data"]
                study_matched_indices.add(best_gold_wrapper["index"])
                all_matched_gold_indices.add(best_gold_wrapper["index"])

                for field in NUMERIC_FIELDS:
                    p_val = pred_row.get(field)
                    g_val = gold_row.get(field)
                    is_p_none = p_val is None
                    is_g_none = g_val is None

                    if is_g_none and is_p_none:
                        # Gold says value unavailable, prediction correctly leaves it empty -> TN
                        tn += 1
                        per_field_counts[field]["tn"] += 1
                    elif is_g_none and not is_p_none:
                        # Gold says unavailable, prediction supplied a value -> FP (hallucination)
                        fp += 1
                        per_field_counts[field]["fp"] += 1
                    elif not is_g_none and is_p_none:
                        # Gold has value, prediction missing -> FN
                        fn += 1
                        per_field_counts[field]["fn"] += 1
                    else:
                        if compare_numbers(p_val, g_val):
                            tp += 1
                            per_field_counts[field]["tp"] += 1
                        else:
                            # Gold has value, prediction has value but incorrect -> FN
                            fn += 1
                            per_field_counts[field]["fn"] += 1
            else:
                for field in NUMERIC_FIELDS:
                    if pred_row.get(field) is not None:
                        fp += 1
                        per_field_counts[field]["fp"] += 1

    total_gold_rows = len(gold_data)
    for i in range(total_gold_rows):
        if i not in all_matched_gold_indices:
            gold_row = gold_data[i]
            for field in NUMERIC_FIELDS:
                g_val = gold_row.get(field)
                if g_val is not None:
                    fn += 1
                    per_field_counts[field]["fn"] += 1
                else:
                    tn += 1
                    per_field_counts[field]["tn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    per_field_metrics: Dict[str, Dict[str, float]] = {}
    for field, counts in per_field_counts.items():
        tp_f = counts["tp"]
        fp_f = counts["fp"]
        fn_f = counts["fn"]
        precision_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
        recall_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        f1_f = 2 * (precision_f * recall_f) / (precision_f + recall_f) if (precision_f + recall_f) > 0 else 0.0
        per_field_metrics[field] = {
            "tp": tp_f,
            "tn": counts["tn"],
            "fp": fp_f,
            "fn": fn_f,
            "precision": precision_f,
            "recall": recall_f,
            "f1": f1_f,
        }

    if verbose:
        print("\n=== Confusion Matrix (Field Level) ===")
        print(f"True Positives  (TP): {tp:<5} (Correctly extracted)")
        print(f"True Negatives  (TN): {tn:<5} (Correctly identified as unavailable)")
        print(f"False Positives (FP): {fp:<5} (Hallucinations / Data generated when none existed)")
        print(f"False Negatives (FN): {fn:<5} (Missed or Incorrectly extracted)")

        print("\n=== Scores ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\n=== Per-field Scores ===")
        for field in NUMERIC_FIELDS:
            m = per_field_metrics[field]
            print(
                f"{field}: P={m['precision']:.4f} "
                f"R={m['recall']:.4f} "
                f"F1={m['f1']:.4f} "
                f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})"
            )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_field_metrics": per_field_metrics,
    }


def main():
    gold_path = GOLD_PATH_DEFAULT
    pred_path = sys.argv[1] if len(sys.argv) > 1 else None
    if pred_path is None:
        print("Usage: python evaluation.py <predictions.json>")
        sys.exit(1)
    if not os.path.exists(gold_path):
        print(f"Please ensure {gold_path} is available.")
        sys.exit(1)
    evaluate(gold_path, pred_path, verbose=True)


if __name__ == "__main__":
    main()
