from typing import List, Dict
from src.evaluation.matching import match_ico, verify_numerical_values

def calculate_metrics(extractions: List[Dict], gold_standard: List[Dict]) -> Dict[str, float]:
    """
    Calculates Precision, Recall, and F1 based on:
    1. Alignment: Matching extraction to gold standard by outcome name (targeted extraction).
    2. Verification: Ensuring numerical values match.
    """

    true_positives = 0
    # False Positives = Items extracted that are either (a) not in gold standard OR (b) in gold standard but have wrong values
    false_positives = 0

    # We need to track which extractions have been 'used' to avoid double counting if duplicate extractions exist
    matched_extraction_indices = set()

    # Iterate through Gold Standard items
    for gold in gold_standard:
        # Filter potential candidates by PMCID first to optimize
        candidates = [
            (i, ext) for i, ext in enumerate(extractions)
            if str(ext.get('pmcid')) == str(gold.get('pmcid')) and i not in matched_extraction_indices
        ]

        best_match_idx = -1
        found_alignment = False

        # Try to find the best aligned extraction (by name)
        for idx, ext in candidates:
            if match_ico(ext, gold):
                # Found an extraction that claims to be this outcome
                found_alignment = True
                best_match_idx = idx
                break  # Assuming one extraction per outcome for now

        if found_alignment:
            matched_extraction_indices.add(best_match_idx)
            aligned_extraction = extractions[best_match_idx]

            # Now VERIFY the values
            if verify_numerical_values(aligned_extraction, gold):
                true_positives += 1
            else:
                # Aligned (found the outcome) but wrong numbers
                # This counts as a False Positive (incorrect extraction)
                false_positives += 1
        # If no alignment found, we simply don't increment TP.
        # FN will be calculated globally based on TP vs Total Gold.

    # Any extractions that were NOT matched to a gold standard item are Spurious / False Positives
    # (e.g., extracting an outcome that wasn't requested or hallucinating one)
    unmatched_extractions = len(extractions) - len(matched_extraction_indices)
    false_positives += unmatched_extractions

    # Robust calculation of False Negatives
    # FN = Everything in Gold Standard that wasn't a True Positive
    false_negatives = len(gold_standard) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
