import os
import json
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR


# Name of the JSON file *inside* each run folder
TARGET_FILENAME = "evaluation_metrics.json"

# Mapping folder substrings to display names in LaTeX
MODEL_MAPPING = {
    "gpt": "GPT-5.2",
    "gemini": "Gemini-3-Pro",
    "claude": "Claude Opus 4.5"
}

SETTING_MAPPING = {
    "zero-shot": "Zero-Shot",
    "few-shot": "Few-Shot"
}

def parse_folder_name(folder_name):
    name_lower = folder_name.lower()
    
    found_model = None
    for key, display_name in MODEL_MAPPING.items():
        if f"_{key}_" in name_lower:
            found_model = display_name
            break
            
    found_setting = None
    for key, display_name in SETTING_MAPPING.items():
        if key in name_lower:
            found_setting = display_name
            break
            
    return found_model, found_setting

def aggregate_results(root_dir):
    aggregated_data = {}
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return {}

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Scanning {len(subdirs)} folders in {root_dir}...")

    for folder in subdirs:
        if "_TEST" not in folder:
            continue

        model, setting = parse_folder_name(folder)
        
        if model and setting:
            if model not in aggregated_data:
                aggregated_data[model] = {}
            
            file_path = os.path.join(root_dir, folder, TARGET_FILENAME)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        aggregated_data[model][setting] = data
                except Exception as e:
                    print(f"  Error loading {folder}: {e}")

    return aggregated_data

def format_metric(metrics, key, is_percent=False, is_best=False):
    """
    Formats a metric value. Bolds it if is_best is True.
    """
    val = metrics.get(key, 0)
    lower = metrics.get(f"{key}_ci_lower")
    upper = metrics.get(f"{key}_ci_upper")
    
    if is_percent:
        val *= 100
        if lower is not None: lower *= 100
        if upper is not None: upper *= 100
    
    # Base string construction
    if lower is not None and upper is not None:
        # Use scriptsize for CI to save space
        val_str = f"{val:.1f} \\scriptsize{{[{lower:.1f}, {upper:.1f}]}}"
    else:
        val_str = f"{val:.1f}"
        
    # Apply bolding
    if is_best:
        return f"\\textbf{{{val_str}}}"
    return val_str

def get_metric_value(results_data, model, setting, json_key, metric_name, data_source="main"):
    """Helper to safely retrieve a raw float value for comparison."""
    run_data = results_data.get(model, {}).get(setting, {})
    
    if data_source == "figures":
        target_root = run_data.get("figures_subset", {})
    else:
        target_root = run_data
    
    if not target_root: return None

    if json_key == "aggregated":
        metrics = target_root.get("aggregated", {})
    else:
        metrics = target_root.get("by_field", {}).get(json_key, {})
    
    return metrics.get(metric_name)

def generate_latex_tables(results_data):
    available_models = sorted(results_data.keys())
    
    # Define the rows
    field_map = [
        ("Total", "aggregated"),
        ("Intervention Mean", "intervention_mean"),
        ("Intervention SD", "intervention_standard_deviation"),
        ("Intervention Group Size", "intervention_group_size"),
        ("Intervention Events", "intervention_events"),
        ("Comparator Mean", "comparator_mean"),
        ("Comparator SD", "comparator_standard_deviation"),
        ("Comparator Group Size", "comparator_group_size"),
        ("Comparator Events", "comparator_events")
    ]

    # =========================================================
    # TABLE 1: HEAD-TO-HEAD (Zero-Shot)
    # Compares all models on F1 and RMSE
    # =========================================================
    print("\n" + "%"*20 + " TABLE 1: ZERO-SHOT COMPARISON " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Head-to-Head Comparison (Zero-Shot). Best scores in bold.}")
    print(r"\label{tab:head_to_head}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{l l c c}")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{Model} & \textbf{F1 [95\% CI]} & \textbf{RMSE [95\% CI]} \\")
    print(r"\midrule")

    setting = "Zero-Shot"

    for display_name, json_key in field_map:
        print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{{display_name}}}}}")
        
        # 1. Find Bests for this specific row (Zero-Shot only)
        best_f1 = -1
        best_rmse = float('inf')
        
        for m in available_models:
            f1 = get_metric_value(results_data, m, setting, json_key, "f1")
            rmse = get_metric_value(results_data, m, setting, json_key, "rmse")
            if f1 is not None and f1 > best_f1: best_f1 = f1
            if rmse is not None and rmse > 0 and rmse < best_rmse: best_rmse = rmse

        # 2. Print Rows
        for model in available_models:
            run_data = results_data.get(model, {}).get(setting, {})
            metrics = {}
            
            # Extract metrics dictionary safely
            if run_data:
                if json_key == "aggregated":
                    metrics = run_data.get("aggregated", {})
                else:
                    metrics = run_data.get("by_field", {}).get(json_key, {})

            if not metrics:
                print(f" & {model} & - & - \\\\")
                continue

            # Check Bests
            val_f1 = metrics.get("f1", 0)
            is_best_f1 = math.isclose(val_f1, best_f1, rel_tol=1e-4)
            
            val_rmse = metrics.get("rmse", 0)
            is_best_rmse = (val_rmse > 0 and math.isclose(val_rmse, best_rmse, rel_tol=1e-4))

            # Format
            f1_str = format_metric(metrics, "f1", is_percent=True, is_best=is_best_f1)
            rmse_str = format_metric(metrics, "rmse", is_percent=False, is_best=is_best_rmse)
            
            print(f" & {model} & {f1_str} & {rmse_str} \\\\")
        
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # =========================================================
    # TABLE 2: STRATEGY ANALYSIS (Gemini Only)
    # Compares Zero-Shot vs Few-Shot
    # =========================================================
    target_model = "Gemini-3-Pro" # Change this if you want to analyze a different model
    
    print("\n" + "%"*20 + " TABLE 2: STRATEGY ANALYSIS " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(f"\\caption{{Effect of Prompting Strategy on {target_model}.}}")
    print(r"\label{tab:strategy_analysis}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{\textbf{F1 Score}} & \multicolumn{2}{c}{\textbf{RMSE}} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r"\textbf{Category} & \textbf{Zero-Shot} & \textbf{Few-Shot} & \textbf{Zero-Shot} & \textbf{Few-Shot} \\")
    print(r"\midrule")

    for display_name, json_key in field_map:
        row_str = f"\\textbf{{{display_name}}} "
        
        # 1. Find Best for this row (Zero vs Few)
        f1_zs = get_metric_value(results_data, target_model, "Zero-Shot", json_key, "f1")
        f1_fs = get_metric_value(results_data, target_model, "Few-Shot", json_key, "f1")
        
        rmse_zs = get_metric_value(results_data, target_model, "Zero-Shot", json_key, "rmse")
        rmse_fs = get_metric_value(results_data, target_model, "Few-Shot", json_key, "rmse")

        # Determine winners
        # (Handle None safely)
        f1_zs = f1_zs if f1_zs else 0
        f1_fs = f1_fs if f1_fs else 0
        rmse_zs = rmse_zs if rmse_zs else float('inf')
        rmse_fs = rmse_fs if rmse_fs else float('inf')

        best_f1 = max(f1_zs, f1_fs)
        best_rmse = min(rmse_zs, rmse_fs)

        # 2. Format Columns
        # Zero-Shot F1
        val_str = f"{f1_zs*100:.1f}"
        if math.isclose(f1_zs, best_f1, rel_tol=1e-4) and f1_zs > 0:
            val_str = f"\\textbf{{{val_str}}}"
        row_str += f"& {val_str} "

        # Few-Shot F1
        val_str = f"{f1_fs*100:.1f}"
        if math.isclose(f1_fs, best_f1, rel_tol=1e-4) and f1_fs > 0:
            val_str = f"\\textbf{{{val_str}}}"
        row_str += f"& {val_str} "

        # Zero-Shot RMSE
        if rmse_zs == float('inf') or rmse_zs == 0:
            val_str = "-"
        else:
            val_str = f"{rmse_zs:.1f}"
            if math.isclose(rmse_zs, best_rmse, rel_tol=1e-4):
                val_str = f"\\textbf{{{val_str}}}"
        row_str += f"& {val_str} "

        # Few-Shot RMSE
        if rmse_fs == float('inf') or rmse_fs == 0:
            val_str = "-"
        else:
            val_str = f"{rmse_fs:.1f}"
            if math.isclose(rmse_fs, best_rmse, rel_tol=1e-4):
                val_str = f"\\textbf{{{val_str}}}"
        row_str += f"& {val_str} \\\\"

        print(row_str)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    if os.path.exists(RESULTS_DIR):
        full_data = aggregate_results(RESULTS_DIR)
        if full_data:
            generate_latex_tables(full_data)
        else:
            print("No matching '_TEST' folders found.")
    else:
        print(f"Directory not found: {RESULTS_DIR}")