from typing import List, Dict
from src.evaluation.matching import match_ico

def calculate_metrics(extractions: List[Dict], gold_standard: List[Dict]) -> Dict[str, float]:
    """
    Calculates Precision, Recall, and potentially other metrics (RMSE for continuous values).
    """
    # This is a simplified evaluation logic.
    # Real evaluation would need to handle multiple extractions vs multiple gold standards (many-to-many).

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_indices = set()

    for extracted in extractions:
        match_found = False
        for idx, gold in enumerate(gold_standard):
            if idx in matched_indices:
                continue

            if match_ico(extracted, gold):
                true_positives += 1
                matched_indices.add(idx)
                match_found = True

                # Check numerical accuracy here if needed (e.g. for RMSE)
                break

        if not match_found:
            false_positives += 1

    false_negatives = len(gold_standard) - len(matched_indices)

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
