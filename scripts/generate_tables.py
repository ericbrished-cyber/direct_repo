import os
import json
import math
import glob
from typing import List, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR
from src.evaluation.metrics import Evaluator

# Path to your Gold Standard
GOLD_STANDARD_PATH = os.path.join(str(Path(__file__).resolve().parents[1]), "data", "gold_standard_clean.json")

# Name of the JSON file *inside* each run folder
TARGET_FILENAME = "evaluation_metrics.json"

MODEL_MAPPING = {
    "gpt": "GPT-5.2",
    "gemini": "Gemini-3-Pro",
    "claude": "Claude Haiku 4.5"
}

SETTING_MAPPING = {
    "zero-shot": "Zero-Shot",
    "few-shot": "Few-Shot"
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_gold_standard():
    if os.path.exists(GOLD_STANDARD_PATH):
        return load_json(GOLD_STANDARD_PATH)
    else:
        print(f"Warning: Gold Standard not found at {GOLD_STANDARD_PATH}")
        return []

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
    run_paths = {} # store paths to raw files for paired calc
    
    if not os.path.exists(root_dir):
        return {}, {}

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Scanning {len(subdirs)} folders in {root_dir}...")

    for folder in subdirs:
        if "_TEST" not in folder: continue

        model, setting = parse_folder_name(folder)
        if model and setting:
            if model not in aggregated_data:
                aggregated_data[model] = {}
                run_paths[model] = {}
            
            folder_path = os.path.join(root_dir, folder)
            file_path = os.path.join(folder_path, TARGET_FILENAME)
            
            # Store metrics
            if os.path.exists(file_path):
                try:
                    aggregated_data[model][setting] = load_json(file_path)
                except Exception as e:
                    print(f"Error loading metrics for {folder}: {e}")
            
            # Store path for raw data loading
            run_paths[model][setting] = folder_path

    return aggregated_data, run_paths

def load_raw_extractions(folder_path):
    all_extractions = []
    files = glob.glob(os.path.join(folder_path, "*.json"))
    for fpath in files:
        fname = os.path.basename(fpath)
        if fname in ["evaluation_metrics.json", "run_metadata.json"]:
            continue

        try:
            data = load_json(fpath)
            pmcid = fname.replace(".json", "")

            # Accept both singular and plural keys
            records = []
            if isinstance(data, dict):
                if "extraction" in data and isinstance(data["extraction"], list):
                    records = data["extraction"]
                elif "extractions" in data and isinstance(data["extractions"], list):
                    records = data["extractions"]
            elif isinstance(data, list):
                records = data

            for item in records:
                if "pmcid" not in item:
                    item["pmcid"] = pmcid
                all_extractions.append(item)
        except Exception:
            pass
    return all_extractions


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
    if not run_data: return None
    if data_source == "figures": target_root = run_data.get("figures_subset", {})
    else: target_root = run_data
        
    if json_key == "aggregated": metrics = target_root.get("aggregated", {})
    else: metrics = target_root.get("by_field", {}).get(json_key, {})
    return metrics.get(metric_name)

def generate_latex_tables(results_data, run_paths):
    available_models = sorted(results_data.keys())
    gold_standard = load_gold_standard()
    
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

    # ... [Same Table 1 Code as before] ...
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

    # ... [Same Table 3 Code] ...
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
        
        val = run_data.get("exact_match_stratified", {}).get("binary", 0)
        txt = f"{val*100:.1f}\\%"
        row_str += f"& \\textbf{{{txt}}} " if math.isclose(val, bests["binary"], rel_tol=1e-4) and val > 0 else f"& {txt} "
        
        val = run_data.get("exact_match_stratified", {}).get("continuous", 0)
        txt = f"{val*100:.1f}\\%"
        row_str += f"& \\textbf{{{txt}}} " if math.isclose(val, bests["continuous"], rel_tol=1e-4) and val > 0 else f"& {txt} "
        
        val = run_data.get("exact_match", 0)
        txt = f"{val*100:.1f}\\%"
        row_str += f"& \\textbf{{{txt}}} \\\\" if math.isclose(val, bests["total"], rel_tol=1e-4) and val > 0 else f"& {txt} \\\\"
        print(row_str)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # -------------------------------------------------------------------------
    # TABLE: BASELINE + DIFFERENCE (Paired Bootstrap - F1 ONLY)
    # -------------------------------------------------------------------------
    print("\n" + "%"*20 + " TABLE: BASELINE + DIFFERENCE (Paired F1) " + "%"*20 + "\n")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Impact of Few-Shot prompting relative to the Zero-Shot baseline. The \textbf{Difference} column ($\Delta$) shows the change in F1-score when adding few-shot examples ($\text{FS} - \text{ZS}$), along with 95\% bootstrap confidence intervals. A negative $\Delta$ indicates that the Zero-Shot prompt performed better.}")
    print(r"\label{tab:zero_vs_few_diff}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{8pt}")
    print(r"\begin{tabular}{l c c}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Baseline F1 (ZS)} & \textbf{Effect of Few-Shot ($\Delta$)} \\")
    print(r" & \textbf{Score} & \textbf{[95\% CI]} \\")
    print(r"\midrule")

    for model in available_models:
        paths = run_paths.get(model, {})
        zs_path = paths.get("Zero-Shot")
        fs_path = paths.get("Few-Shot")
        
        if not zs_path or not fs_path:
            continue
            
        # Load raw data and initialize Evaluators
        zs_raw = load_raw_extractions(zs_path)
        fs_raw = load_raw_extractions(fs_path)
        
        eval_zs = Evaluator(gold_standard, zs_raw)
        eval_fs = Evaluator(gold_standard, fs_raw)
        
        stats_zs = eval_zs.get_bootstrap_source_data()
        stats_fs = eval_fs.get_bootstrap_source_data()
        
        # Get Baseline F1
        zs_metrics = results_data[model]["Zero-Shot"].get("aggregated", {})
        baseline_val = zs_metrics.get("f1", 0) * 100
        
        # Calculate Paired Diff for F1
        point, low, high = Evaluator.compute_paired_difference_ci(stats_zs, stats_fs, metric='f1')
        
        point *= 100
        low *= 100
        high *= 100
        
        print(f"{model} & {baseline_val:.1f} & {point:.1f} \\scriptsize{{[{low:.1f}, {high:.1f}]}} \\\\")
            
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    if os.path.exists(RESULTS_DIR):
        full_data, run_paths = aggregate_results(RESULTS_DIR)
        if full_data:
            generate_latex_tables(full_data, run_paths)
        else:
            print("No matching folders found.")
    else:
        print(f"Directory not found: {RESULTS_DIR}")