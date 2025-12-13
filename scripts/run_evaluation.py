import json
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR, GOLD_STANDARD_PATH
from src.evaluation.metrics import calculate_metrics 

def load_run_data(run_folder_name):
    run_path = RESULTS_DIR / run_folder_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run folder not found: {run_path}")

    all_extractions = []
    files = list(run_path.glob("*.json"))
    print(f"Scanning {len(files)} files in {run_path}...")

    for file_path in files:
        if file_path.name in ["run_metadata.json", "evaluation_metrics.json", "final_results.json"]:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "extraction" in data and isinstance(data["extraction"], list):
                all_extractions.extend(data["extraction"])
        except Exception as e:
            print(f"Skipping corrupt file {file_path.name}: {e}")

    return all_extractions

def format_ci(val, lower, upper):
    if lower == 0 and upper == 0:
        return f"{val:.2%}"
    return f"{val:.2%} [{lower:.2f}, {upper:.2f}]"

def format_rmse_ci(val, lower, upper):
    if lower == 0 and upper == 0:
        return f"{val:.4f}"
    return f"{val:.4f} [{lower:.2f}, {upper:.2f}]"

def print_breakdown(title, breakdown_dict):
    if not breakdown_dict:
        return
    print(f"\n--- {title} ---")
    header = f"{'FIELD':<35} | {'F1 (95% CI)':<25} | {'RMSE':<20} | {'PREC':<8}"
    print(header)
    print("-" * len(header))
    
    for field, m in breakdown_dict.items():
        f1_str = format_ci(m.get('f1', 0), m.get('f1_ci_lower', 0), m.get('f1_ci_upper', 0))
        rmse_str = format_rmse_ci(m.get('rmse', 0), m.get('rmse_ci_lower', 0), m.get('rmse_ci_upper', 0))
        print(f"{field:<35} | {f1_str:<25} | {rmse_str:<20} | {m['precision']:.2f}")

def run_evaluation_task(run_folder, split):
    print("Step 1: Compiling extracted data...")
    extractions = load_run_data(run_folder)
    print(f"Loaded {len(extractions)} extracted items.")

    if not extractions:
        print("Warning: No extractions found.")
        return

    print("Step 2: Loading Gold Standard...")
    if not GOLD_STANDARD_PATH.exists():
        print(f"Error: Gold standard not found at {GOLD_STANDARD_PATH}")
        return

    with open(GOLD_STANDARD_PATH, 'r', encoding='utf-8') as f:
        full_gold = json.load(f)
    
    gold_standard = [item for item in full_gold if item.get("split") == split]
    print(f"Found {len(gold_standard)} Gold Standard items for split '{split}'.")

    if not gold_standard:
        print(f"Error: No gold standard data found for split '{split}'.")
        return

    print("Step 3: Calculating metrics (includes bootstrap for CI)...")
    all_metrics = calculate_metrics(extractions, gold_standard)
    
    agg = all_metrics["aggregated"]
    
    # 4. Output Results
    print("\n" + "="*80)
    print(f"EVALUATION REPORT: {run_folder}")
    print("="*80)
    
    # --- AGGREGATED METRICS ---
    print(f"{'METRIC':<25} {'VALUE (95% CI)':<30}")
    print("-" * 80)
    
    f1_str = format_ci(agg['f1'], agg.get('f1_ci_lower',0), agg.get('f1_ci_upper',0))
    rmse_str = format_rmse_ci(agg['rmse'], agg.get('rmse_ci_lower',0), agg.get('rmse_ci_upper',0))
    
    print(f"{'Precision':<25} {agg['precision']:.2%}")
    print(f"{'Recall':<25} {agg['recall']:.2%}")
    print(f"{'F1 Score':<25} {f1_str}")
    print(f"{'RMSE':<25} {rmse_str}")
    print(f"{'Exact Match':<25} {agg['exact_match']:.2%}")
    print("-" * 80)
    print(f"True Positives: {agg.get('true_positives', 0)}")
    print(f"False Positives:{agg.get('false_positives', 0)}")
    print(f"False Negatives:{agg.get('false_negatives', 0)}")
    print("="*80)

    # --- FIELD BREAKDOWN (Overall) ---
    print_breakdown("OVERALL BREAKDOWN BY FIELD TYPE", all_metrics.get("by_field", {}))

    # --- FIGURE SUBSET ---
    fig_data = all_metrics.get("figures_subset", {})
    if fig_data:
        print("\n\n" + "="*80)
        print("FIGURE DATA SUBSET ANALYSIS")
        print("="*80)
        
        fig_agg = fig_data.get("aggregated", {})
        fig_f1_str = format_ci(fig_agg.get('f1', 0), fig_agg.get('f1_ci_lower',0), fig_agg.get('f1_ci_upper',0))
        fig_rmse_str = format_rmse_ci(fig_agg.get('rmse', 0), fig_agg.get('rmse_ci_lower',0), fig_agg.get('rmse_ci_upper',0))
        
        print(f"{'Precision':<25} {fig_agg.get('precision', 0):.2%}")
        print(f"{'Recall':<25} {fig_agg.get('recall', 0):.2%}")
        print(f"{'F1 Score':<25} {fig_f1_str}")
        print(f"{'RMSE':<25} {fig_rmse_str}")
        print(f"Support (Items):          {fig_agg.get('true_positives',0) + fig_agg.get('false_negatives',0)}")
        
        # --- FIELD BREAKDOWN (Figures) ---
        print_breakdown("FIGURE SUBSET BREAKDOWN BY FIELD", fig_data.get("by_field", {}))
        print("="*80)

    # Save metrics
    save_path = RESULTS_DIR / run_folder / "evaluation_metrics.json"
    with open(save_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evaluation on extracted results")
    parser.add_argument("--run_folder", type=str, required=True, help="Folder name in data/results/")
    parser.add_argument("--split", type=str, default="DEV", help="DEV or TEST")
    args = parser.parse_args()

    run_evaluation_task(args.run_folder, args.split)