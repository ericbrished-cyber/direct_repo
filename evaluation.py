"""
Evaluation utilities with comprehensive metrics.

Implements:
- Precision, Recall, F1-score (field-level and ICO-level)
- Mean Squared Error (MSE) for numeric deviations
- Exact Match accuracy (all fields correct for an ICO)
"""

from __future__ import annotations

import difflib
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

# Default gold path (single source of truth)
GOLD_PATH_DEFAULT = "gold-standard/gold_standard_clean.json"

# === Configuration ===
MATCH_THRESHOLD = 0.8  # Similarity threshold for ICO matching (outcome/intervention/comparator)
NUMBER_TOLERANCE = 0.01  # Relative tolerance for numeric comparison (1%)

# All numeric fields we evaluate
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

# Field sets for different outcome types (used in exact match calculation)
BINARY_OUTCOME_FIELDS = [
    "intervention_group_size",
    "comparator_group_size",
    "intervention_events",
    "comparator_events",
]

CONTINUOUS_OUTCOME_FIELDS = [
    "intervention_group_size",
    "comparator_group_size",
    "intervention_mean",
    "comparator_mean",
    "intervention_standard_deviation",
    "comparator_standard_deviation",
]


def load_json_file(filepath: str) -> Any:
    """
    Load a JSON file from disk.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Parsed JSON data (dict or list)
        
    Raises:
        SystemExit if file not found or invalid JSON
    """
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
    """
    Normalize text for comparison: lowercase, strip whitespace, collapse spaces.
    
    Example: "  Multiple   Spaces  " -> "multiple spaces"
    
    Args:
        text: Input text (can be any type, non-strings return empty string)
        
    Returns:
        Normalized string
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase, strip leading/trailing whitespace, collapse internal whitespace
    return " ".join(text.lower().strip().split())


def calculate_similarity(pred_row: Dict[str, Any], gold_row: Dict[str, Any]) -> float:
    """
    Calculate similarity between prediction and gold ICO rows.
    
    Uses SequenceMatcher on concatenated outcome|intervention|comparator strings.
    This allows fuzzy matching since exact text matches are rare.
    
    Example:
        pred: "mortality | drug A | placebo"
        gold: "mortality rate | Drug A | placebo"
        -> high similarity score (>0.8)
    
    Args:
        pred_row: Predicted ICO row
        gold_row: Gold standard ICO row
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Concatenate the three key fields with a separator
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
    
    # SequenceMatcher gives ratio of matching characters/subsequences
    return difflib.SequenceMatcher(None, pred_str, gold_str).ratio()


def compare_numbers(pred_val: Any, gold_val: Any) -> bool:
    """
    Check if two numbers match within relative tolerance.
    
    Uses math.isclose with NUMBER_TOLERANCE (1% by default).
    Handles string inputs by parsing them to float.
    
    Example:
        compare_numbers(100, 101) with 1% tolerance -> False
        compare_numbers(100, 100.5) with 1% tolerance -> True
    
    Args:
        pred_val: Predicted numeric value (can be string or number)
        gold_val: Gold standard numeric value (can be string or number)
        
    Returns:
        True if values match within tolerance, False otherwise
    """
    try:
        # Convert strings to float, handling comma separators
        if isinstance(pred_val, str):
            pred_val = float(pred_val.replace(",", ""))
        if isinstance(gold_val, str):
            gold_val = float(gold_val.replace(",", ""))
        
        # Use math.isclose for floating-point comparison with relative tolerance
        return math.isclose(float(pred_val), float(gold_val), rel_tol=NUMBER_TOLERANCE)
    except (ValueError, TypeError):
        # If conversion fails, values don't match
        return False


def calculate_mse(pred_val: Any, gold_val: Any) -> float:
    """
    Calculate squared error between predicted and gold values.
    
    This is the component of Mean Squared Error for a single field.
    Large deviations have quadratically larger impact on MSE.
    
    Args:
        pred_val: Predicted numeric value
        gold_val: Gold standard numeric value
        
    Returns:
        Squared error, or 0.0 if conversion fails
    """
    try:
        # Convert to float, handling string inputs
        if isinstance(pred_val, str):
            pred_val = float(pred_val.replace(",", ""))
        if isinstance(gold_val, str):
            gold_val = float(gold_val.replace(",", ""))
        
        # Calculate squared difference
        diff = float(pred_val) - float(gold_val)
        return diff ** 2
    except (ValueError, TypeError):
        # If conversion fails, return 0 (no contribution to MSE)
        return 0.0


def check_exact_match(pred_row: Dict[str, Any], gold_row: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if a predicted ICO row exactly matches the gold standard.
    
    Exact match criteria:
    - For binary outcomes: all 4 fields must match (group sizes + events)
    - For continuous outcomes: all 6 fields must match (group sizes + means + SDs)
    
    This is a stricter criterion than field-level matching used in precision/recall.
    
    Args:
        pred_row: Predicted ICO row
        gold_row: Gold standard ICO row
        
    Returns:
        Tuple of (is_exact_match, outcome_type)
        where outcome_type is "binary", "continuous", or "unknown"
    """
    # Determine outcome type by checking which fields are present in gold
    has_events = gold_row.get("intervention_events") is not None or \
                 gold_row.get("comparator_events") is not None
    has_means = gold_row.get("intervention_mean") is not None or \
                gold_row.get("comparator_mean") is not None
    
    # Classify outcome type
    if has_events and not has_means:
        outcome_type = "binary"
        fields_to_check = BINARY_OUTCOME_FIELDS
    elif has_means and not has_events:
        outcome_type = "continuous"
        fields_to_check = CONTINUOUS_OUTCOME_FIELDS
    else:
        # Mixed or unclear - shouldn't happen in well-formed data
        outcome_type = "unknown"
        fields_to_check = NUMERIC_FIELDS
    
    # Check all required fields match
    all_match = True
    for field in fields_to_check:
        pred_val = pred_row.get(field)
        gold_val = gold_row.get(field)
        
        # Both must be non-null and match
        if pred_val is None or gold_val is None:
            all_match = False
            break
        
        if not compare_numbers(pred_val, gold_val):
            all_match = False
            break
    
    return all_match, outcome_type


def _group_gold_by_pmcid(gold_rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Group gold standard rows by PMCID for efficient lookup.
    
    Each entry includes the original data and its index for tracking matches.
    
    Args:
        gold_rows: List of gold standard ICO entries
        
    Returns:
        Dictionary mapping PMCID -> list of {"data": row, "index": i}
    """
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for idx, entry in enumerate(gold_rows):
        pmcid = entry.get("pmcid")
        grouped.setdefault(pmcid, []).append({"data": entry, "index": idx})
    return grouped


def _normalize_pmcid(value: Any) -> Any:
    """
    Normalize PMCID to integer if possible, otherwise keep as-is.
    
    This handles both integer and string PMCIDs consistently.
    
    Args:
        value: PMCID value (any type)
        
    Returns:
        Integer PMCID if convertible, otherwise original value
    """
    try:
        return int(value)
    except Exception:
        return value


def _extract_rows(pred_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract ICO rows from a prediction object.
    
    Handles multiple possible field names for backward compatibility.
    
    Args:
        pred_obj: Prediction object (dictionary)
        
    Returns:
        List of ICO rows
    """
    return pred_obj.get("rows") or pred_obj.get("extractions") or []


def _group_predictions(pred_data: Any) -> List[Dict[str, Any]]:
    """
    Normalize predictions into standard format: list of {"pmcid": X, "rows": [...]}.
    
    Handles three input formats:
    1. Single dict with 'extractions' field (current format)
    2. Flat list of ICO rows (groups by PMCID)
    3. List of dicts with 'rows' or 'extractions' fields
    
    Args:
        pred_data: Raw prediction data (dict or list)
        
    Returns:
        List of normalized prediction objects
        
    Raises:
        ValueError if format is unrecognized
    """
    # Format 1: Single dict with rows/extractions
    if isinstance(pred_data, dict):
        rows = _extract_rows(pred_data)
        pmcid = _normalize_pmcid(
            pred_data.get("pmcid") or (rows[0].get("pmcid") if rows else None)
        )
        return [{"pmcid": pmcid, "rows": rows}]
    
    # Format 2: Flat list of ICO rows - group by PMCID
    if isinstance(pred_data, list) and pred_data and \
       "rows" not in pred_data[0] and "extractions" not in pred_data[0]:
        grouped: Dict[Any, Dict[str, Any]] = {}
        for row in pred_data:
            pid = _normalize_pmcid(row.get("pmcid"))
            grouped.setdefault(pid, {"pmcid": pid, "rows": []})
            grouped[pid]["rows"].append(row)
        return list(grouped.values())
    
    # Format 3: List of grouped objects
    if isinstance(pred_data, list):
        grouped_objs = []
        for obj in pred_data:
            rows = _extract_rows(obj)
            pmcid = _normalize_pmcid(
                obj.get("pmcid") or (rows[0].get("pmcid") if rows else None)
            )
            grouped_objs.append({"pmcid": pmcid, "rows": rows})
        return grouped_objs
    
    raise ValueError("Unsupported prediction format")


def evaluate(gold_file: str, pred_file: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of predictions against gold standard.
    
    Computes:
    1. Field-level metrics: TP, TN, FP, FN, Precision, Recall, F1
    2. MSE (Mean Squared Error) for numeric deviations
    3. Exact match accuracy (all fields correct for an ICO)
    
    Matching algorithm:
    - ICO rows matched by fuzzy similarity on outcome/intervention/comparator
    - Fields within matched ICOs compared numerically with tolerance
    - Unmatched predictions count as FP, unmatched gold entries as FN
    
    Args:
        gold_file: Path to gold standard JSON
        pred_file: Path to predictions JSON
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Load data files
    gold_data_all = load_json_file(gold_file)
    pred_data = load_json_file(pred_file)
    
    # Normalize predictions to standard format
    grouped_predictions = _group_predictions(pred_data)
    
    # Filter gold standard to only PMCIDs present in predictions
    pmcid_filter = {pred_obj.get("pmcid") for pred_obj in grouped_predictions}
    gold_data = [row for row in gold_data_all if row.get("pmcid") in pmcid_filter]
    
    # Group gold data by PMCID for efficient lookup
    gold_by_pmcid = _group_gold_by_pmcid(gold_data)
    
    # Track which gold ICOs have been matched (to identify FNs later)
    all_matched_gold_indices = set()
    
    # Initialize counters for confusion matrix
    tp = tn = fp = fn = 0
    
    # Per-field confusion matrix counters
    per_field_counts: Dict[str, Dict[str, int]] = {
        field: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for field in NUMERIC_FIELDS
    }
    
    # MSE calculation: sum of squared errors and count of comparisons
    mse_sum = 0.0
    mse_count = 0
    per_field_mse: Dict[str, Dict[str, float]] = {
        field: {"sum": 0.0, "count": 0} for field in NUMERIC_FIELDS
    }
    
    # Exact match tracking
    exact_match_correct = 0
    exact_match_total = 0
    exact_match_by_type: Dict[str, Dict[str, int]] = {
        "binary": {"correct": 0, "total": 0},
        "continuous": {"correct": 0, "total": 0},
    }
    
    if verbose:
        print(f"--- Evaluation ({gold_file} vs {pred_file}) ---")
    
    # ===== MAIN EVALUATION LOOP =====
    # Process each prediction object (one per PMCID)
    for pred_obj in grouped_predictions:
        pmcid = pred_obj.get("pmcid")
        rows = pred_obj.get("rows") or []
        
        # If PMCID not in gold standard, all predictions are false positives
        if pmcid not in gold_by_pmcid:
            for row in rows:
                for field in NUMERIC_FIELDS:
                    if row.get(field) is not None:
                        fp += 1
                        per_field_counts[field]["fp"] += 1
            continue
        
        # Get gold ICOs for this PMCID
        gold_candidates = gold_by_pmcid[pmcid]
        
        # Track which gold ICOs have been matched within this study
        study_matched_indices = set()
        
        # Process each predicted ICO row
        for pred_row in rows:
            # Find best matching gold ICO using fuzzy similarity
            best_score = -1.0
            best_gold_wrapper = None
            
            for wrapper in gold_candidates:
                # Skip if this gold ICO already matched to another prediction
                if wrapper["index"] in study_matched_indices:
                    continue
                
                # Calculate similarity score
                score = calculate_similarity(pred_row, wrapper["data"])
                
                # Track best match
                if score > best_score:
                    best_score = score
                    best_gold_wrapper = wrapper
            
            # Check if match is above threshold
            if best_score >= MATCH_THRESHOLD and best_gold_wrapper:
                # ===== MATCHED ICO: Compare fields =====
                gold_row = best_gold_wrapper["data"]
                
                # Mark this gold ICO as matched
                study_matched_indices.add(best_gold_wrapper["index"])
                all_matched_gold_indices.add(best_gold_wrapper["index"])
                
                # Check exact match for this ICO
                is_exact, outcome_type = check_exact_match(pred_row, gold_row)
                if outcome_type in ["binary", "continuous"]:
                    exact_match_total += 1
                    exact_match_by_type[outcome_type]["total"] += 1
                    if is_exact:
                        exact_match_correct += 1
                        exact_match_by_type[outcome_type]["correct"] += 1
                
                # Compare each numeric field
                for field in NUMERIC_FIELDS:
                    p_val = pred_row.get(field)
                    g_val = gold_row.get(field)
                    is_p_none = p_val is None
                    is_g_none = g_val is None
                    
                    # Case analysis based on field availability
                    if is_g_none and is_p_none:
                        # Gold says unavailable, prediction agrees -> TRUE NEGATIVE
                        # This is correct identification of missing data
                        tn += 1
                        per_field_counts[field]["tn"] += 1
                        
                    elif is_g_none and not is_p_none:
                        # Gold says unavailable, but prediction provided value -> FALSE POSITIVE
                        # This is a hallucination - model invented data
                        fp += 1
                        per_field_counts[field]["fp"] += 1
                        
                    elif not is_g_none and is_p_none:
                        # Gold has value, but prediction missing -> FALSE NEGATIVE
                        # Model failed to extract available data
                        fn += 1
                        per_field_counts[field]["fn"] += 1
                        
                    else:
                        # Both have values - check if they match numerically
                        if compare_numbers(p_val, g_val):
                            # Values match within tolerance -> TRUE POSITIVE
                            tp += 1
                            per_field_counts[field]["tp"] += 1
                        else:
                            # Values don't match -> FALSE NEGATIVE
                            # (incorrect extraction counts as missing the correct value)
                            fn += 1
                            per_field_counts[field]["fn"] += 1
                        
                        # Calculate MSE contribution (only for both non-null)
                        se = calculate_mse(p_val, g_val)
                        mse_sum += se
                        mse_count += 1
                        per_field_mse[field]["sum"] += se
                        per_field_mse[field]["count"] += 1
            else:
                # ===== UNMATCHED PREDICTION: All fields are false positives =====
                # No gold ICO matched this prediction (similarity too low)
                for field in NUMERIC_FIELDS:
                    if pred_row.get(field) is not None:
                        fp += 1
                        per_field_counts[field]["fp"] += 1
    
    # ===== HANDLE UNMATCHED GOLD ICOS =====
    # Any gold ICO that wasn't matched represents missed extractions
    total_gold_rows = len(gold_data)
    for i in range(total_gold_rows):
        if i not in all_matched_gold_indices:
            gold_row = gold_data[i]
            
            # Each field in the unmatched gold ICO
            for field in NUMERIC_FIELDS:
                g_val = gold_row.get(field)
                if g_val is not None:
                    # Gold has value, but no prediction matched -> FALSE NEGATIVE
                    fn += 1
                    per_field_counts[field]["fn"] += 1
                else:
                    # Gold says unavailable, no prediction -> TRUE NEGATIVE
                    # (correctly didn't hallucinate for this missing ICO)
                    tn += 1
                    per_field_counts[field]["tn"] += 1
    
    # ===== CALCULATE AGGREGATE METRICS =====
    
    # Overall precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Mean Squared Error
    mse = mse_sum / mse_count if mse_count > 0 else 0.0
    rmse = math.sqrt(mse) if mse > 0 else 0.0
    
    # Exact match accuracy
    exact_match_accuracy = exact_match_correct / exact_match_total if exact_match_total > 0 else 0.0
    
    # Per-field metrics (precision, recall, F1, MSE)
    per_field_metrics: Dict[str, Dict[str, float]] = {}
    for field, counts in per_field_counts.items():
        tp_f = counts["tp"]
        fp_f = counts["fp"]
        fn_f = counts["fn"]
        
        # Field-level precision, recall, F1
        precision_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
        recall_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        f1_f = (
            2 * (precision_f * recall_f) / (precision_f + recall_f)
            if (precision_f + recall_f) > 0
            else 0.0
        )
        
        # Field-level MSE
        mse_f = (
            per_field_mse[field]["sum"] / per_field_mse[field]["count"]
            if per_field_mse[field]["count"] > 0
            else 0.0
        )
        rmse_f = math.sqrt(mse_f) if mse_f > 0 else 0.0
        
        per_field_metrics[field] = {
            "tp": tp_f,
            "tn": counts["tn"],
            "fp": fp_f,
            "fn": fn_f,
            "precision": precision_f,
            "recall": recall_f,
            "f1": f1_f,
            "mse": mse_f,
            "rmse": rmse_f,
            "num_comparisons": per_field_mse[field]["count"],
        }
    
    # Exact match by outcome type
    exact_match_accuracy_binary = (
        exact_match_by_type["binary"]["correct"] / exact_match_by_type["binary"]["total"]
        if exact_match_by_type["binary"]["total"] > 0
        else 0.0
    )
    exact_match_accuracy_continuous = (
        exact_match_by_type["continuous"]["correct"] / exact_match_by_type["continuous"]["total"]
        if exact_match_by_type["continuous"]["total"] > 0
        else 0.0
    )
    
    # ===== VERBOSE OUTPUT =====
    if verbose:
        print("\n=== Confusion Matrix (Field Level) ===")
        print(f"True Positives  (TP): {tp:<5} (Correctly extracted)")
        print(f"True Negatives  (TN): {tn:<5} (Correctly identified as unavailable)")
        print(f"False Positives (FP): {fp:<5} (Hallucinations / Data generated when none existed)")
        print(f"False Negatives (FN): {fn:<5} (Missed or Incorrectly extracted)")
        
        print("\n=== Overall Scores ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        print("\n=== Error Metrics ===")
        print(f"MSE (Mean Squared Error):      {mse:.4f} ({mse_count} comparisons)")
        print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        
        print("\n=== Exact Match Accuracy ===")
        print(f"Overall:     {exact_match_accuracy:.4f} ({exact_match_correct}/{exact_match_total})")
        print(f"Binary:      {exact_match_accuracy_binary:.4f} "
              f"({exact_match_by_type['binary']['correct']}/{exact_match_by_type['binary']['total']})")
        print(f"Continuous:  {exact_match_accuracy_continuous:.4f} "
              f"({exact_match_by_type['continuous']['correct']}/{exact_match_by_type['continuous']['total']})")
        
        print("\n=== Per-field Scores ===")
        for field in NUMERIC_FIELDS:
            m = per_field_metrics[field]
            print(
                f"{field:40s} "
                f"P={m['precision']:.4f} "
                f"R={m['recall']:.4f} "
                f"F1={m['f1']:.4f} "
                f"MSE={m['mse']:.4f} "
                f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})"
            )
    
    # Return all computed metrics
    return {
        # Confusion matrix
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        
        # Overall metrics
        "precision": precision,
        "recall": recall,
        "f1": f1,
        
        # Error metrics
        "mse": mse,
        "rmse": rmse,
        "num_mse_comparisons": mse_count,
        
        # Exact match metrics
        "exact_match_accuracy": exact_match_accuracy,
        "exact_match_correct": exact_match_correct,
        "exact_match_total": exact_match_total,
        "exact_match_by_type": exact_match_by_type,
        
        # Per-field breakdowns
        "per_field_metrics": per_field_metrics,
    }


def main():
    """
    Command-line interface for evaluation.
    
    Usage: python evaluation.py <predictions.json>
    """
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