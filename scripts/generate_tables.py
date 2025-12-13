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
        val_str = f"{val:.1f} [{lower:.1f}, {upper:.1f}]"
    else:
        val_str = f"{val:.1f}"
        
    # Apply bolding
    if is_best:
        return f"\\textbf{{{val_str}}}"
    return val_str

def get_best_values_for_category(results_data, json_key, data_source, available_models, settings):
    """
    Scans all models to find the best value for each (Setting, Metric) pair.
    Returns a dict: { (setting, metric_key): best_float_value }
    """
    best_map = {}
    
    metrics_to_track = ["precision", "recall", "f1", "rmse"]
    
    # Initialize with worst possible values
    for setting in settings:
        for m in metrics_to_track:
            if m == "rmse":
                best_map[(setting, m)] = float('inf') # Lower is better
            else:
                best_map[(setting, m)] = -1.0         # Higher is better

    for model in available_models:
        for setting in settings:
            run_data = results_data.get(model, {}).get(setting, {})
            
            # Navigate to correct node
            if data_source == "figures":
                target_root = run_data.get("figures_subset", {})
            else:
                target_root = run_data
            
            if not target_root: continue

            if json_key == "aggregated":
                metrics = target_root.get("aggregated", {})
            else:
                metrics = target_root.get("by_field", {}).get(json_key, {})
            
            if not metrics: continue

            # Update Bests
            for m in metrics_to_track:
                val = metrics.get(m, 0)
                current_best = best_map[(setting, m)]
                
                if m == "rmse":
                    if val < current_best and val > 0: # valid RMSE > 0
                        best_map[(setting, m)] = val
                else:
                    if val > current_best:
                        best_map[(setting, m)] = val
                        
    return best_map

def generate_latex_tables(results_data):
    available_models = sorted(results_data.keys())
    settings = ["Zero-Shot", "Few-Shot"]
    
    field_map = [
        ("Total", "aggregated"),
        ("IM", "intervention_mean"),
        ("ISD", "intervention_standard_deviation"),
        ("IGS", "intervention_group_size"),
        ("IE", "intervention_events"),
        ("CM", "comparator_mean"),
        ("CSD", "comparator_standard_deviation"),
        ("CGS", "comparator_group_size"),
        ("CE", "comparator_events")
    ]

    def print_metric_table(table_caption, table_label, data_source="main"):
        print(r"\begin{table}[ht]")
        print(r"\centering")
        print(f"\\caption{{{table_caption}}}")
        print(f"\\label{{{table_label}}}")
        print(r"\tiny") 
        print(r"\setlength{\tabcolsep}{2pt}") 
        print(r"\begin{tabular}{l l c c c c | c c c c}")
        print(r"\toprule")
        print(r" & & \multicolumn{4}{c}{\textbf{Zero-Shot}} & \multicolumn{4}{c}{\textbf{Few-Shot}} \\")
        print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
        print(r"\textbf{Category} & \textbf{Model} & P & R & F1 & RMSE & P & R & F1 & RMSE \\")
        print(r"\midrule")

        for display_name, json_key in field_map:
            print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{{display_name}}}}}")
            
            # --- 1. Find best values for this category ---
            bests = get_best_values_for_category(results_data, json_key, data_source, available_models, settings)
            
            for model in available_models:
                row_str = f" & {model} "
                
                for setting in settings:
                    run_data = results_data.get(model, {}).get(setting, {})
                    
                    if data_source == "figures":
                        target_root = run_data.get("figures_subset", {})
                    else:
                        target_root = run_data

                    if not target_root:
                        row_str += "& - & - & - & - "
                        continue

                    if json_key == "aggregated":
                        metrics = target_root.get("aggregated", {})
                    else:
                        metrics = target_root.get("by_field", {}).get(json_key, {})
                    
                    if not metrics:
                        row_str += "& - & - & - & - "
                        continue

                    # --- 2. Check and Format ---
                    # Precision
                    val_p = metrics.get('precision', 0)
                    is_best_p = math.isclose(val_p, bests[(setting, 'precision')], rel_tol=1e-4)
                    p_str = f"\\textbf{{{val_p*100:.1f}}}" if is_best_p else f"{val_p*100:.1f}"

                    # Recall
                    val_r = metrics.get('recall', 0)
                    is_best_r = math.isclose(val_r, bests[(setting, 'recall')], rel_tol=1e-4)
                    r_str = f"\\textbf{{{val_r*100:.1f}}}" if is_best_r else f"{val_r*100:.1f}"

                    # F1 (with CI)
                    val_f1 = metrics.get('f1', 0)
                    is_best_f1 = math.isclose(val_f1, bests[(setting, 'f1')], rel_tol=1e-4)
                    f1_str = format_metric(metrics, "f1", is_percent=True, is_best=is_best_f1)

                    # RMSE (with CI)
                    val_rmse = metrics.get('rmse', 0)
                    # For RMSE, we check if it matches the MINIMUM
                    is_best_rmse = (val_rmse > 0 and math.isclose(val_rmse, bests[(setting, 'rmse')], rel_tol=1e-4))
                    rmse_str = format_metric(metrics, "rmse", is_percent=False, is_best=is_best_rmse)
                    
                    row_str += f"& {p_str} & {r_str} & {f1_str} & {rmse_str} "
                
                print(row_str + r"\\")
            print(r"\midrule")
        
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # --- 1. Main Results Table ---
    print("\n" + "%"*20 + " TABLE 1: MAIN METRICS " + "%"*20 + "\n")
    print_metric_table(
        table_caption="Performance metrics on the full TEST set. Best values per column in bold.",
        table_label="tab:main_results",
        data_source="main"
    )

    # --- 2. Figure Subset Table ---
    print("\n" + "%"*20 + " TABLE 2: FIGURE SUBSET " + "%"*20 + "\n")
    print_metric_table(
        table_caption="Performance on data extracted from Figures. Best values per column in bold.",
        table_label="tab:figure_results",
        data_source="figures"
    )

    # --- 3. Exact Match Table ---
    print("\n" + "%"*20 + " TABLE 3: EXACT MATCH " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Exact Match (EM) accuracy. Best values in bold.}")
    print(r"\label{tab:exact_match}")
    print(r"\small")
    print(r"\begin{tabular}{l l c c}")
    print(r"\toprule")
    print(r"\textbf{Outcome Type} & \textbf{Model} & \textbf{EM (Zero-Shot)} & \textbf{EM (Few-Shot)} \\")
    print(r"\midrule")
    
    # Calculate Bests for Exact Match
    em_bests = {}
    for s in settings:
        best_val = -1
        for m in available_models:
            val = results_data.get(m, {}).get(s, {}).get("exact_match", 0)
            if val > best_val: best_val = val
        em_bests[s] = best_val

    print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{Overall}}}}")
    for model in available_models:
        row_str = f" & {model} "
        for setting in settings:
            val = results_data.get(model, {}).get(setting, {}).get("exact_match", 0)
            val_pct = val * 100
            
            if math.isclose(val, em_bests[setting], rel_tol=1e-4):
                row_str += f"& \\textbf{{{val_pct:.1f}}}\\% "
            else:
                row_str += f"& {val_pct:.1f}\\% "
        print(row_str + r"\\")
    
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
