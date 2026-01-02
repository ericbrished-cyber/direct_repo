import json
import random
import re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np

# --- CONFIGURATION ---
GOLD_PATH = "data/gold_standard_clean.json"
RESULTS_DIR = "data/results/20251215_131614_claude-haiku_zero-shot_TEST" 
OUTPUT_FILE = "side_by_side_comparison_haiku_ZERO.txt"
SAMPLE_SIZE = 20

MATCH_FIELDS = [
    ("intervention_mean", "fuzzy"),
    ("comparator_mean", "fuzzy"),
    ("intervention_sd", "fuzzy"),
    ("comparator_sd", "fuzzy"),
    ("intervention_group_size", "exact"),
    ("comparator_group_size", "exact"),
    ("intervention_events", "exact"),
    ("comparator_events", "exact"),
]

SD_KEYS = {
    "intervention": [
        "intervention_standard_deviation",
        "intervention_sd",
        "intervention_se",
        "intervention_standard_error",
    ],
    "comparator": [
        "comparator_standard_deviation",
        "comparator_sd",
        "comparator_se",
        "comparator_standard_error",
    ],
}

def similarity(a, b):
    """Calculates string similarity (0.0 to 1.0)"""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def get_best_match(gold_item, predictions):
    """
    Finds the single best prediction that matches the gold item
    based on Outcome + Intervention + Comparator text.
    """
    if not predictions: return None
    
    # Create a signature for the gold item
    gold_sig = f"{gold_item.get('outcome', '')} {gold_item.get('intervention', '')} {gold_item.get('comparator', '')}"
    
    best_match = None
    best_score = 0.0
    
    for pred in predictions:
        # Create signature for prediction
        pred_sig = f"{pred.get('outcome', '')} {pred.get('intervention', '')} {pred.get('comparator', '')}"
        
        score = similarity(gold_sig, pred_sig)
        if score > best_score:
            best_score = score
            best_match = pred
            
    # Threshold: If similarity is too low, assume the model missed it entirely
    if best_score < 0.6: 
        return None
        
    return best_match

def clean_val(val):
    return str(val) if val is not None else "-"

def get_sd_display(item, prefix):
    # Check all possible keys for SD/SE
    keys = [f'{prefix}_standard_deviation', f'{prefix}_sd', f'{prefix}_se', f'{prefix}_standard_error']
    for k in keys:
        if item.get(k) is not None:
            return str(item.get(k))
    return "-"

def get_sd_value(item, prefix):
    for key in SD_KEYS[prefix]:
        val = item.get(key)
        if val is not None and str(val).strip() != "":
            return val
    return None

def get_field_value(item, field):
    if not item:
        return None
    if field == "intervention_sd":
        return get_sd_value(item, "intervention")
    if field == "comparator_sd":
        return get_sd_value(item, "comparator")
    return item.get(field)

def parse_numeric(val):
    if val is None:
        return None
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned in {"", "-"}:
            return None
        val = cleaned
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def values_match(val1, val2, match_type, tolerance=0.001):
    v1 = parse_numeric(val1)
    v2 = parse_numeric(val2)
    if v1 is None and v2 is None:
        return True
    if v1 is None or v2 is None:
        return False
    if match_type == "exact":
        return v1 == v2
    return np.isclose(v1, v2, rtol=tolerance)

def items_match(gold_item, pred_item, tolerance=0.001):
    if not pred_item:
        return False
    for field, match_type in MATCH_FIELDS:
        gold_val = get_field_value(gold_item, field)
        pred_val = get_field_value(pred_item, field)
        if not values_match(gold_val, pred_val, match_type, tolerance=tolerance):
            return False
    return True

def report_has_error(gold_items, preds, tolerance=0.001):
    for gold_item in gold_items:
        best_pred = get_best_match(gold_item, preds)
        if not items_match(gold_item, best_pred, tolerance=tolerance):
            return True
    return False

def load_predictions(file_path):
    with open(file_path, 'r') as f:
        res_json = json.load(f)

    pmcid = str(res_json.get('pmcid', file_path.stem))
    raw_text = res_json.get('raw_text', '')

    try:
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        preds = json.loads(clean_text)
        if isinstance(preds, dict):
            preds = preds.get('extractions', [])
    except:
        preds = []

    return pmcid, preds

def generate_block(tag, item):
    """Generates the text block for a single item (Gold or Pred)"""
    if not item:
        return f">>> {tag}:\n    (No matching extraction found)\n"

    out = f">>> {tag}:\n"
    out += f"    Outcome:      {item.get('outcome', 'N/A')}\n"
    out += f"    Intervention: {item.get('intervention', 'N/A')}\n"
    out += f"    Comparator:   {item.get('comparator', 'N/A')}\n"
    out += "    " + "-"*64 + "\n"
    out += "    METRIC        | INTERVENTION                | COMPARATOR\n"
    out += "    " + "-"*64 + "\n"

    # 1. Continuous Data
    i_mean = clean_val(item.get('intervention_mean'))
    c_mean = clean_val(item.get('comparator_mean'))
    i_sd = get_sd_display(item, 'intervention')
    c_sd = get_sd_display(item, 'comparator')

    if i_mean != "-" or c_mean != "-" or i_sd != "-" or c_sd != "-":
        out += f"    Mean (SD)     | {i_mean:<8} ({i_sd:<8})   | {c_mean:<8} ({c_sd:<8})\n"

    # 2. Binary Data
    i_events = clean_val(item.get('intervention_events'))
    c_events = clean_val(item.get('comparator_events'))
    if i_events != "-" or c_events != "-":
        out += f"    Events        | {i_events:<23} | {c_events}\n"

    # 3. Group Size
    i_n = clean_val(item.get('intervention_group_size'))
    c_n = clean_val(item.get('comparator_group_size'))
    out += f"    Group Size    | {i_n:<23} | {c_n}\n"
    
    return out

def run_analysis():
    print("Loading Gold Standard...")
    with open(GOLD_PATH, 'r') as f:
        gold_data = json.load(f)
    
    # Group Gold by PMCID
    gold_map = {}
    for item in gold_data:
        pmcid = str(item.get('pmcid'))
        if pmcid not in gold_map: gold_map[pmcid] = []
        gold_map[pmcid].append(item)

    print("Scanning Results...")
    results_path = Path(RESULTS_DIR)
    all_files = [f for f in results_path.glob("*.json") if "run_metadata" not in f.name]
    
    # Filter only files that have corresponding gold data
    valid_files = [f for f in all_files if str(json.load(open(f)).get('pmcid', f.stem)) in gold_map]
    
    print(f"Found {len(valid_files)} papers with Gold Standard data.")

    error_files = []
    for file_path in valid_files:
        pmcid, preds = load_predictions(file_path)
        gold_items = gold_map.get(pmcid, [])
        if not gold_items:
            continue
        if report_has_error(gold_items, preds):
            error_files.append((file_path, pmcid, preds, gold_items))

    if not error_files:
        print("No error cases found to sample.")
        return

    print(f"Sampling from {len(error_files)} papers with errors.")
    sampled_files = random.sample(error_files, min(SAMPLE_SIZE, len(error_files)))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("QUALITATIVE ERROR ANALYSIS: PAIRED COMPARISON\n")
        out.write("=============================================\n\n")

        for i, (file_path, pmcid, preds, gold_items) in enumerate(sampled_files, 1):

            out.write(f"PAPER {i}: PMCID {pmcid}\n")
            out.write("-" * 80 + "\n")

            # Iterate through GOLD items to drive the comparison
            for idx, gold_item in enumerate(gold_items, 1):
                best_pred = get_best_match(gold_item, preds)
                
                # Check for mismatch to highlight errors
                is_perfect = items_match(gold_item, best_pred)

                header = f"PAIR {idx}: {gold_item.get('outcome')}"
                if is_perfect: header += " [MATCH]"
                else: header += " [MISMATCH/MISSING]"
                
                out.write(f"{header}\n")
                
                # Print Gold
                out.write(generate_block("GOLD STANDARD", gold_item))
                out.write("\n")
                
                # Print Prediction
                out.write(generate_block("PREDICTION", best_pred))
                out.write("\n\n")
            
            out.write("="*80 + "\n\n")

    print(f"Comparison generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_analysis()
