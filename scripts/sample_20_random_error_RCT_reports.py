import json
import random
import re
from pathlib import Path
from difflib import SequenceMatcher

# --- CONFIGURATION ---
GOLD_PATH = "data/gold_standard_clean.json"
RESULTS_DIR = "data/results/GPTTEST/20251213_002444_gpt_zero-shot_TEST" 
OUTPUT_FILE = "side_by_side_comparison.txt"
SAMPLE_SIZE = 20

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
    sampled_files = random.sample(valid_files, min(SAMPLE_SIZE, len(valid_files)))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("QUALITATIVE ERROR ANALYSIS: PAIRED COMPARISON\n")
        out.write("=============================================\n\n")

        for i, file_path in enumerate(sampled_files, 1):
            # Load Prediction
            with open(file_path, 'r') as f:
                res_json = json.load(f)
            
            pmcid = str(res_json.get('pmcid', file_path.stem))
            raw_text = res_json.get('raw_text', '')
            
            try:
                # Clean and parse JSON
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                preds = json.loads(clean_text)
                if isinstance(preds, dict): preds = preds.get('extractions', [])
            except:
                preds = []

            gold_items = gold_map.get(pmcid, [])

            out.write(f"PAPER {i}: PMCID {pmcid}\n")
            out.write("-" * 80 + "\n")

            # Iterate through GOLD items to drive the comparison
            for idx, gold_item in enumerate(gold_items, 1):
                best_pred = get_best_match(gold_item, preds)
                
                # Check for mismatch to highlight errors
                is_perfect = False
                if best_pred:
                    # Simple equality check for flagging (optional)
                    if (str(gold_item.get('intervention_mean')) == str(best_pred.get('intervention_mean')) and
                        str(gold_item.get('intervention_group_size')) == str(best_pred.get('intervention_group_size'))):
                        is_perfect = True

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