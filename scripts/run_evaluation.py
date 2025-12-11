import json
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR, GOLD_STANDARD_PATH
# This imports the wrapper function or class from your updated metrics file
from src.evaluation.metrics import calculate_metrics 

def load_run_data(run_folder_name):
    """
    Compiles all JSON files in the run folder into a single list of extractions.
    """
    run_path = RESULTS_DIR / run_folder_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run folder not found: {run_path}")

    all_extractions = []
    
    # Iterate over all .json files (excluding metadata/metrics files)
    files = list(run_path.glob("*.json"))
    print(f"Scanning {len(files)} files in {run_path}...")

    for file_path in files:
        if file_path.name in ["run_metadata.json", "evaluation_metrics.json", "final_results.json"]:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # The extraction list already has PMCIDs injected by run_extraction.py
            if "extraction" in data and isinstance(data["extraction"], list):
                all_extractions.extend(data["extraction"])
                
        except Exception as e:
            print(f"Skipping corrupt file {file_path.name}: {e}")

    return all_extractions

def run_evaluation_task(run_folder, split):
    # 1. Load Extractions
    print("Step 1: Compiling extracted data...")
    extractions = load_run_data(run_folder)
    print(f"Loaded {len(extractions)} extracted items.")

    if not extractions:
        print("Warning: No extractions found. Check the run folder or if extraction failed.")
        return

    # 2. Load Gold Standard
    print("Step 2: Loading Gold Standard...")
    if not GOLD_STANDARD_PATH.exists():
        print(f"Error: Gold standard not found at {GOLD_STANDARD_PATH}")
        return

    with open(GOLD_STANDARD_PATH, 'r', encoding='utf-8') as f:
        full_gold = json.load(f)
    
    # Filter Gold Standard by the requested split
    gold_standard = [item for item in full_gold if item.get("split") == split]
    print(f"Found {len(gold_standard)} Gold Standard items for split '{split}'.")

    if not gold_standard:
        print(f"Error: No gold standard data found for split '{split}'.")
        return

    # 3. Calculate Metrics
    print("Step 3: Calculating metrics (Thesis Definitions)...")
    
    # This calls your updated metrics logic
    metrics = calculate_metrics(extractions, gold_standard)

    # 4. Print & Save Results
    print("\n" + "="*40)
    print(f"EVALUATION REPORT: {run_folder}")
    print("="*40)
    print(f"Precision:      {metrics.get('precision', 0):.2%}")
    print(f"Recall:         {metrics.get('recall', 0):.2%}")
    print(f"F1 Score:       {metrics.get('f1', 0):.2%}")
    print("-" * 40)
    print(f"RMSE:           {metrics.get('rmse', 0):.4f}")
    print(f"Exact Match:    {metrics.get('exact_match', 0):.2%}")
    print("-" * 40)
    print(f"True Positives: {metrics.get('true_positives', 0)}")
    print(f"False Positives:{metrics.get('false_positives', 0)}")
    print(f"False Negatives:{metrics.get('false_negatives', 0)}")
    print("="*40)

    # Save metrics to file
    save_path = RESULTS_DIR / run_folder / "evaluation_metrics.json"
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation on extracted results")
    parser.add_argument("--run_folder", type=str, required=True, help="Folder name in data/results/ (e.g. 20251010_gpt_zero-shot_DEV)")
    parser.add_argument("--split", type=str, default="DEV", help="DEV or TEST")
    args = parser.parse_args()

    run_evaluation_task(args.run_folder, args.split)