from typing import Any, Dict, Optional, Union
import math
import difflib

def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Returns a ratio of similarity between two strings (0.0 to 1.0).
    """
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()

def is_numerical_match(val1: Optional[Union[float, int, str]], val2: Optional[Union[float, int, str]], tol: float = 1e-3) -> bool:
    """
    Checks if two values match numerically.
    Handles None/null correctly (both must be None).
    """
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False

    try:
        f1 = float(val1)
        f2 = float(val2)
        return math.isclose(f1, f2, rel_tol=tol)
    except (ValueError, TypeError):
        return False

def verify_numerical_values(extracted: Dict, gold: Dict) -> bool:
    """
    Verifies that all numerical fields in the extracted data match the gold standard.
    This assumes the extracted record has already been aligned to the gold record by outcome name.
    """
    numerical_fields = [
        # Binary
        "intervention_events",
        "intervention_group_size",
        "comparator_events",
        "comparator_group_size",
        # Continuous
        "intervention_mean",
        "intervention_standard_deviation",
        "comparator_mean",
        "comparator_standard_deviation"
    ]

    for field in numerical_fields:
        gold_val = gold.get(field)
        extracted_val = extracted.get(field)

        if not is_numerical_match(extracted_val, gold_val):
            return False

    return True

def match_ico(extracted: Dict, ground_truth: Dict, threshold: float = 0.8) -> bool:
    """
    Legacy matching function.
    In the new evaluation flow, this is used primarily for ALIGNMENT (identifying which extraction belongs to which target),
    not for the final 'True Positive' verification.
    """
    outcome_score = fuzzy_match_score(extracted.get('outcome', ''), ground_truth.get('outcome', ''))
    return outcome_score >= threshold
