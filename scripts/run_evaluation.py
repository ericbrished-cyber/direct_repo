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
    
    figure_gold_count = sum(1 for item in gold_standard if item.get("is_data_in_figure_graphics"))

    # 3. Calculate Metrics
    print("Step 3: Calculating metrics...")
    all_metrics = calculate_metrics(extractions, gold_standard)
    figure_metrics = calculate_metrics(extractions, gold_standard, subset="figures")
    
    # Extract the parts
    agg = all_metrics["aggregated"]
    by_field = all_metrics["by_field"]
    fig_agg = figure_metrics["aggregated"]
    fig_by_field = figure_metrics["by_field"]

    # 4. Output Results
    print("\n" + "="*70)
    print(f"EVALUATION REPORT: {run_folder}")
    print("="*70)
    print(f"{'METRIC':<25} {'VALUE':<10} {'95% CI':<20}")
    print("-" * 70)
    print(f"{'Precision':<25} {agg['precision']:.2%}")
    print(f"{'Recall':<25} {agg['recall']:.2%}")
    
    # Updated F1 Line
    f1_ci = f"[{agg.get('f1_ci_lower',0):.2f}, {agg.get('f1_ci_upper',0):.2f}]"
    print(f"{'F1 Score':<25} {agg['f1']:.2%}      {f1_ci}")
    
    # Updated RMSE Line
    rmse_ci = f"[{agg.get('rmse_ci_lower',0):.2f}, {agg.get('rmse_ci_upper',0):.2f}]"
    print(f"{'RMSE':<25} {agg['rmse']:.4f}      {rmse_ci}")
    
    print(f"{'Exact Match':<25} {agg['exact_match']:.2%}")
    print("-" * 70)
    print(f"True Positives: {agg.get('true_positives', 0)}")
    print(f"False Positives:{agg.get('false_positives', 0)}")
    print(f"False Negatives:{agg.get('false_negatives', 0)}")
    print("="*70)
    
    print("\n--- BREAKDOWN BY FIELD TYPE ---")
    header = f"{'FIELD':<35} | {'P':<8} | {'R':<8} | {'F1':<8} | {'RMSE':<8}"
    print(header)
    print("-" * len(header))
    
    for field, m in by_field.items():
        print(f"{field:<35} | {m['precision']:.1%}   | {m['recall']:.1%}   | {m['f1']:.1%}   | {m['rmse']:.2f}")

    print("\n--- FIGURE-ONLY METRICS (is_data_in_figure_graphics=True) ---")
    print(f"Gold rows tagged as figures: {figure_gold_count}")
    print(f"{'METRIC':<20} {'AGGREGATED':<10}")
    print("-" * 60)
    print(f"{'Precision':<20} {fig_agg['precision']:.2%}")
    print(f"{'Recall':<20} {fig_agg['recall']:.2%}")
    print(f"{'F1 Score':<20} {fig_agg['f1']:.2%}")
    print(f"{'RMSE':<20} {fig_agg['rmse']:.4f}")
    print(f"{'Exact Match':<20} {fig_agg['exact_match']:.2%}")
    print("-" * 60)

    print("\n--- FIGURE-ONLY BREAKDOWN BY FIELD TYPE ---")
    print(header)
    print("-" * len(header))
    for field, m in fig_by_field.items():
        print(f"{field:<35} | {m['precision']:.1%}   | {m['recall']:.1%}   | {m['f1']:.1%}   | {m['rmse']:.2f}")

    print("="*60)

    # Save metrics to file
    # Keep overall metrics at the top level for backward compatibility and
    # tuck figure-only metrics under a new key in the same file.
    all_metrics["figure_subset"] = figure_metrics
    save_path = RESULTS_DIR / run_folder / "evaluation_metrics.json"
    with open(save_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation on extracted results")
    parser.add_argument("--run_folder", type=str, required=True, help="Folder name in data/results/ (e.g. 20251010_gpt_zero-shot_DEV)")
    parser.add_argument("--split", type=str, default="DEV", help="DEV or TEST")
    args = parser.parse_args()

    run_evaluation_task(args.run_folder, args.split)
