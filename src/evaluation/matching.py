from typing import Any, Dict
import difflib

def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Returns a ratio of similarity between two strings (0.0 to 1.0).
    """
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()

def match_ico(extracted: Dict, ground_truth: Dict, threshold: float = 0.8) -> bool:
    """
    Determines if an extracted ICO matches the ground truth ICO.
    We might require matching on outcome name, intervention, and comparator.
    """
    outcome_score = fuzzy_match_score(extracted.get('outcome', ''), ground_truth.get('outcome', ''))

    # You might want simpler or stricter logic here.
    # For now, let's say primary match is on Outcome string.
    return outcome_score >= threshold
