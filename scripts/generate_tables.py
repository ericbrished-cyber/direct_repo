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
    # Sort models to ensure consistent order (e.g. Claude, Gemini, GPT)
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

    # -------------------------------------------------------------------------
    # TABLE 1: HEAD-TO-HEAD (Zero-Shot) - Standard P/R/F1/RMSE table
    # -------------------------------------------------------------------------
    print("\n" + "%"*20 + " TABLE 1: ZERO-SHOT COMPARISON " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Zero-Shot Performance: Head-to-Head Comparison.}")
    print(r"\label{tab:main_results}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\begin{tabular}{l l c c c c}")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{Model} & \textbf{Prec} & \textbf{Rec} & \textbf{F1 [95\% CI]} & \textbf{RMSE [95\% CI]} \\")
    print(r"\midrule")

    for display_name, json_key in field_map:
        print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{{display_name}}}}}")
        
        best_vals = {"precision": -1, "recall": -1, "f1": -1, "rmse": float('inf')}
        setting = "Zero-Shot"
        
        # Determine Bests
        for m in available_models:
            run_data = results_data.get(m, {}).get(setting, {})
            for metric in best_vals.keys():
                val = get_metric_value(run_data, json_key, metric, "main")
                if val is not None:
                    if metric == "rmse":
                        if val > 0 and val < best_vals[metric]: best_vals[metric] = val
                    else:
                        if val > best_vals[metric]: best_vals[metric] = val

        for model in available_models:
            run_data = results_data.get(model, {}).get(setting, {})
            metrics = run_data.get("aggregated", {}) if json_key == "aggregated" else run_data.get("by_field", {}).get(json_key, {})
            
            if not metrics:
                print(f" & {model} & - & - & - & - \\\\")
                continue

            row_str = f" & {model} "
            p_val = metrics.get('precision', 0)
            row_str += f"& {format_metric(metrics, 'precision', True, math.isclose(p_val, best_vals['precision'], rel_tol=1e-4))} "
            r_val = metrics.get('recall', 0)
            row_str += f"& {format_metric(metrics, 'recall', True, math.isclose(r_val, best_vals['recall'], rel_tol=1e-4))} "
            f1_val = metrics.get('f1', 0)
            row_str += f"& {format_metric(metrics, 'f1', True, math.isclose(f1_val, best_vals['f1'], rel_tol=1e-4))} "
            rmse_val = metrics.get('rmse', 0)
            is_best_rmse = (rmse_val > 0 and math.isclose(rmse_val, best_vals['rmse'], rel_tol=1e-4))
            row_str += f"& {format_metric(metrics, 'rmse', False, is_best_rmse)} \\\\"
            print(row_str)
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # -------------------------------------------------------------------------
    # TABLE 2: STRATEGY ANALYSIS (ALL MODELS)
    # -------------------------------------------------------------------------
    # Uses all available models since all now have Few-Shot data
    strat_models = available_models 
    
    # Build dynamic column string: "l l | c c | c c | c c" etc.
    col_def = "l l " + "| c c " * len(strat_models)
    
    print("\n" + "%"*20 + " TABLE 2: STRATEGY ANALYSIS (ALL MODELS) " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Impact of Few-Shot Prompting across all models.}")
    print(r"\label{tab:strategy_models}")
    print(r"\scriptsize") # Use scriptsize to fit all columns
    print(r"\setlength{\tabcolsep}{3pt}") # Tight columns
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print(r"\toprule")
    
    # Dynamic Header Row 1: Model Names
    header1 = "& "
    for m in strat_models:
        header1 += f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{m}}}}} "
    print(header1 + r"\\")
    
    # Dynamic Header Row 2: ZS / FS
    header2 = r"\textbf{Category} & \textbf{Metric} "
    for _ in strat_models:
        header2 += r"& \textbf{ZS} & \textbf{FS} "
    print(header2 + r"\\")
    
    print(r"\midrule")

    for display_name, json_key in field_map:
        # We print two rows per category: F1 and RMSE
        for metric_name, label in [("f1", "F1"), ("rmse", "RMSE")]:
            
            # 1. Determine best value in this row (across all models/settings)
            best_in_row = -1.0 if metric_name == "f1" else float('inf')
            
            values_map = {}
            for m in strat_models:
                for s in ["Zero-Shot", "Few-Shot"]:
                    run = results_data.get(m, {}).get(s, {})
                    val = get_metric_value(run, json_key, metric_name)
                    
                    if val is not None:
                        if metric_name == "rmse" and val <= 0: continue
                        values_map[(m, s)] = val
                        
                        if metric_name == "f1":
                            if val > best_in_row: best_in_row = val
                        else:
                            if val < best_in_row: best_in_row = val

            # 2. Build Row String
            if metric_name == "f1":
                row_str = f"\\multirow{{2}}{{*}}{{\\textbf{{{display_name}}}}} & {label} "
            else:
                row_str = f" & {label} "

            for m in strat_models:
                for s in ["Zero-Shot", "Few-Shot"]:
                    val = values_map.get((m, s))
                    
                    if val is None:
                        row_str += "& - "
                    else:
                        is_percent = (metric_name == "f1")
                        display_val = val * 100 if is_percent else val
                        
                        # Check bolding
                        is_best = False
                        if metric_name == "f1" and best_in_row > 0:
                            is_best = math.isclose(val, best_in_row, rel_tol=1e-4)
                        elif metric_name == "rmse" and best_in_row < float('inf'):
                            is_best = math.isclose(val, best_in_row, rel_tol=1e-4)

                        if is_best:
                            row_str += f"& \\textbf{{{display_val:.1f}}} "
                        else:
                            row_str += f"& {display_val:.1f} "
            
            print(row_str + r"\\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- TABLE 3: EXACT MATCH (Zero-Shot) ---
    print("\n" + "%"*20 + " TABLE 3: EXACT MATCH " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Exact Match (EM) accuracy on Zero-Shot, stratified by Outcome Type.}")
    print(r"\label{tab:exact_match}")
    print(r"\small")
    print(r"\begin{tabular}{l c c c}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Binary} & \textbf{Continuous} & \textbf{Total} \\")
    print(r"\midrule")
    
    setting = "Zero-Shot"
    bests = {"binary": -1, "continuous": -1, "total": -1}
    for m in available_models:
        run_data = results_data.get(m, {}).get(setting, {})
        for k in bests:
            if k == "total":
                val = run_data.get("exact_match", 0)
            else:
                val = run_data.get("exact_match_stratified", {}).get(k, 0)
            if val > bests[k]: bests[k] = val

    for model in available_models:
        run_data = results_data.get(model, {}).get(setting, {})
        row_str = f"{model} "
        
        # Binary
        val = run_data.get("exact_match_stratified", {}).get("binary", 0)
        txt = f"{val*100:.1f}\\%"
        row_str += f"& \\textbf{{{txt}}} " if math.isclose(val, bests["binary"], rel_tol=1e-4) and val > 0 else f"& {txt} "
        
        # Continuous
        val = run_data.get("exact_match_stratified", {}).get("continuous", 0)
        txt = f"{val*100:.1f}\\%"
        row_str += f"& \\textbf{{{txt}}} " if math.isclose(val, bests["continuous"], rel_tol=1e-4) and val > 0 else f"& {txt} "
        
        # Total
        val = run_data.get("exact_match", 0)
        txt = f"{val*100:.1f}\\%"
        row_str += f"& \\textbf{{{txt}}} \\\\" if math.isclose(val, bests["total"], rel_tol=1e-4) and val > 0 else f"& {txt} \\\\"
        print(row_str)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- TABLE 4: FIGURE SUBSET (Zero-Shot) ---
    print_comparison_table(
        "Table 4: Figure Data Subset",
        "Performance on Figure-sourced data (Zero-Shot).",
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