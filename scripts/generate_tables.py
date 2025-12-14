import os
import json
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR


# Name of the JSON file *inside* each run folder
TARGET_FILENAME = "evaluation_metrics.json"

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
        return {}

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Scanning {len(subdirs)} folders in {root_dir}...")

    for folder in subdirs:
        if "_TEST" not in folder: continue

        model, setting = parse_folder_name(folder)
        if model and setting:
            if model not in aggregated_data:
                aggregated_data[model] = {}
            
            file_path = os.path.join(root_dir, folder, TARGET_FILENAME)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        aggregated_data[model][setting] = json.load(f)
                except Exception as e:
                    print(f"Error: {e}")

    return aggregated_data

def format_metric(metrics, key, is_percent=False, is_best=False):
    val = metrics.get(key, 0)
    lower = metrics.get(f"{key}_ci_lower")
    upper = metrics.get(f"{key}_ci_upper")
    
    if is_percent:
        val *= 100
        if lower is not None: lower *= 100
        if upper is not None: upper *= 100
    
    if lower is not None and upper is not None:
        val_str = f"{val:.1f} \\scriptsize{{[{lower:.1f}, {upper:.1f}]}}"
    else:
        val_str = f"{val:.1f}"
        
    if is_best:
        return f"\\textbf{{{val_str}}}"
    return val_str

def get_metric_value(run_data, json_key, metric_name, data_source="main"):
    """Helper to retrieve raw float values."""
    if not run_data: return None
    
    if data_source == "figures":
        target_root = run_data.get("figures_subset", {})
    else:
        target_root = run_data
        
    if json_key == "aggregated":
        metrics = target_root.get("aggregated", {})
    else:
        metrics = target_root.get("by_field", {}).get(json_key, {})
        
    return metrics.get(metric_name)

def generate_latex_tables(results_data):
    available_models = sorted(results_data.keys())
    
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

    # Helper to generate standard comparison tables
    def print_comparison_table(title, caption, label, setting, data_source="main"):
        print("\n" + "%"*20 + f" {title.upper()} " + "%"*20 + "\n")
        print(r"\begin{table}[ht]")
        print(r"\centering")
        print(f"\\caption{{{caption}}}")
        print(f"\\label{{{label}}}")
        print(r"\small")
        print(r"\setlength{\tabcolsep}{4pt}")
        print(r"\begin{tabular}{l l c c}")
        print(r"\toprule")
        print(r"\textbf{Category} & \textbf{Model} & \textbf{F1 [95\% CI]} & \textbf{RMSE [95\% CI]} \\")
        print(r"\midrule")

        for display_name, json_key in field_map:
            print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{{display_name}}}}}")
            
            # Find bests
            best_f1 = -1
            best_rmse = float('inf')
            
            for m in available_models:
                run_data = results_data.get(m, {}).get(setting, {})
                f1 = get_metric_value(run_data, json_key, "f1", data_source)
                rmse = get_metric_value(run_data, json_key, "rmse", data_source)
                
                if f1 is not None and f1 > best_f1: best_f1 = f1
                if rmse is not None and rmse > 0 and rmse < best_rmse: best_rmse = rmse

            for model in available_models:
                run_data = results_data.get(model, {}).get(setting, {})
                
                # Navigate to specific metrics dict
                if data_source == "figures":
                    root = run_data.get("figures_subset", {})
                else:
                    root = run_data
                
                if json_key == "aggregated":
                    metrics = root.get("aggregated", {}) if root else {}
                else:
                    metrics = root.get("by_field", {}).get(json_key, {}) if root else {}

                if not metrics:
                    print(f" & {model} & - & - \\\\")
                    continue

                # Format
                val_f1 = metrics.get("f1", 0)
                is_best_f1 = math.isclose(val_f1, best_f1, rel_tol=1e-4)
                f1_str = format_metric(metrics, "f1", is_percent=True, is_best=is_best_f1)
                
                val_rmse = metrics.get("rmse", 0)
                # RMSE best logic: must be > 0 and close to min
                is_best_rmse = (val_rmse > 0 and math.isclose(val_rmse, best_rmse, rel_tol=1e-4))
                rmse_str = format_metric(metrics, "rmse", is_percent=False, is_best=is_best_rmse)
                
                print(f" & {model} & {f1_str} & {rmse_str} \\\\")
            print(r"\midrule")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # --- TABLE 1: MAIN RESULTS (Zero-Shot) ---
    print_comparison_table(
        "Table 1: Main Head-to-Head",
        "Performance metrics on the full TEST set (Zero-Shot). Best scores in bold.",
        "tab:main_results",
        "Zero-Shot",
        data_source="main"
    )

    # --- TABLE 2: STRATEGY ANALYSIS (Gemini Only) ---
    target_model = "Gemini-3-Pro"
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
        # Get raw values
        run_zs = results_data.get(target_model, {}).get("Zero-Shot", {})
        run_fs = results_data.get(target_model, {}).get("Few-Shot", {})
        
        f1_zs = get_metric_value(run_zs, json_key, "f1")
        f1_fs = get_metric_value(run_fs, json_key, "f1")
        rmse_zs = get_metric_value(run_zs, json_key, "rmse")
        rmse_fs = get_metric_value(run_fs, json_key, "rmse")

        # Comparisons
        f1_zs = f1_zs if f1_zs else 0
        f1_fs = f1_fs if f1_fs else 0
        best_f1 = max(f1_zs, f1_fs)
        
        rmse_zs = rmse_zs if rmse_zs else float('inf')
        rmse_fs = rmse_fs if rmse_fs else float('inf')
        best_rmse = min(rmse_zs, rmse_fs)

        # Print Row
        row = f"\\textbf{{{display_name}}} "
        
        # F1 Columns
        row += f"& \\textbf{{{f1_zs*100:.1f}}}" if math.isclose(f1_zs, best_f1) and f1_zs > 0 else f"& {f1_zs*100:.1f}"
        row += f"& \\textbf{{{f1_fs*100:.1f}}}" if math.isclose(f1_fs, best_f1) and f1_fs > 0 else f"& {f1_fs*100:.1f}"
        
        # RMSE Columns
        if rmse_zs == float('inf'): row += " & -"
        else: row += f"& \\textbf{{{rmse_zs:.1f}}}" if math.isclose(rmse_zs, best_rmse) else f"& {rmse_zs:.1f}"
            
        if rmse_fs == float('inf'): row += " & - \\\\"
        else: row += f"& \\textbf{{{rmse_fs:.1f}}}" if math.isclose(rmse_fs, best_rmse) else f"& {rmse_fs:.1f} \\\\"
        
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- TABLE 3: EXACT MATCH ---
    print("\n" + "%"*20 + " TABLE 3: EXACT MATCH " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Exact Match (EM) accuracy at the ICO level. Comparison of all models and settings.}")
    print(r"\label{tab:exact_match}")
    print(r"\begin{tabular}{l c c}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{EM (Zero-Shot)} & \textbf{EM (Few-Shot)} \\")
    print(r"\midrule")
    
    settings = ["Zero-Shot", "Few-Shot"]
    # Find global best EM
    global_best = -1
    for m in available_models:
        for s in settings:
            val = results_data.get(m, {}).get(s, {}).get("exact_match", 0)
            if val > global_best: global_best = val

    for model in available_models:
        row_str = f"{model} "
        for setting in settings:
            val = results_data.get(model, {}).get(setting, {}).get("exact_match", 0)
            val_pct = val * 100
            
            if math.isclose(val, global_best, rel_tol=1e-4) and val > 0:
                row_str += f"& \\textbf{{{val_pct:.1f}}}\\% "
            else:
                if val == 0: row_str += "& - "
                else: row_str += f"& {val_pct:.1f}\\% "
        print(row_str + r"\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- TABLE 4: FIGURE SUBSET (Zero-Shot) ---
    print_comparison_table(
        "Table 4: Figure Data Subset",
        "Performance on data extracted purely from Figures/Graphs (Zero-Shot).",
        "tab:figure_subset",
        "Zero-Shot",
        data_source="figures"
    )

if __name__ == "__main__":
    if os.path.exists(RESULTS_DIR):
        full_data = aggregate_results(RESULTS_DIR)
        if full_data:
            generate_latex_tables(full_data)
        else:
            print("No matching folders found.")
    else:
        print(f"Directory not found: {RESULTS_DIR}")