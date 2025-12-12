import os
import json
import re

from src.config import RESULTS_DIR

# Name of the JSON file *inside* each run folder
TARGET_FILENAME = "evaluation_metrics" 

# Mapping folder substrings to display names in LaTeX
MODEL_MAPPING = {
    "gpt": "GPT-5.1",
    "gemini": "Gemini-3-Pro",
    "claude": "Claude Opus 4.5"
}

SETTING_MAPPING = {
    "zero-shot": "Zero-Shot",
    "few-shot": "Few-Shot"
}

def parse_folder_name(folder_name):
    """
    Extracts model name and setting from folder strings.
    """
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
    """
    Walks through the root directory, finds matching folders with '_TEST',
    and builds the master dictionary.
    """
    aggregated_data = {}
    
    try:
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except FileNotFoundError:
        print(f"Error: Directory '{root_dir}' not found.")
        return {}

    print(f"Scanning {len(subdirs)} folders in {root_dir}...")

    for folder in subdirs:
        # Strictly require the folder name to contain "_TEST"
        if "_TEST" not in folder:
            # print(f"  Skipping (not a TEST folder): {folder}") 
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
                        print(f"  Loaded: {model} / {setting} from {folder}")
                except Exception as e:
                    print(f"  Error loading JSON in {folder}: {e}")
            else:
                print(f"  Warning: {TARGET_FILENAME} not found in {folder}")
        else:
            print(f"  Skipping (pattern mismatch): {folder}")

    return aggregated_data

def generate_latex_tables(results_data):
    """
    Generates the LaTeX code based on the aggregated structure.
    """
    available_models = sorted(results_data.keys())
    settings = ["Zero-Shot", "Few-Shot"]
    
    # 1. Field Metrics Table Map
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

    print("\n" + "%" * 20 + " TABLE 1 LATEX CODE " + "%" * 20 + "\n")
    
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Field-level performance metrics (Precision, Recall, F1, RMSE) across categories (TEST Set).}")
    print(r"\label{tab:field_metrics}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{l l c c c c | c c c c}")
    print(r"\toprule")
    print(r" & & \multicolumn{4}{c}{\textbf{Zero-Shot}} & \multicolumn{4}{c}{\textbf{Few-Shot}} \\")
    print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    print(r"\textbf{Category} & \textbf{Model} & P & R & F1 & RMSE & P & R & F1 & RMSE \\")
    print(r"\midrule")

    for display_name, json_key in field_map:
        print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{{display_name}}}}}")
        
        for i, model in enumerate(available_models):
            row_str = f" & {model} "
            
            for setting in settings:
                data_node = results_data.get(model, {}).get(setting, {})
                
                if not data_node:
                    row_str += "& - & - & - & - "
                    continue

                if json_key == "aggregated":
                    metrics = data_node.get("aggregated", {})
                else:
                    metrics = data_node.get("by_field", {}).get(json_key, {})
                
                p = metrics.get("precision", 0)
                r = metrics.get("recall", 0)
                f1 = metrics.get("f1", 0)
                rmse = metrics.get("rmse", 0)
                
                row_str += f"& {p:.2f} & {r:.2f} & {f1:.2f} & {rmse:.2f} "
            
            print(row_str + r"\\")
        
        print(r"\midrule")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "%" * 20 + " TABLE 2 LATEX CODE " + "%" * 20 + "\n")

    # 2. Exact Match Table Map
    subsets = [
        ("Binary (4 fields)", "binary_subset"),
        ("Continuous (6 fields)", "continuous_subset")
    ]

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Exact Match (EM) accuracy at the ICO level (TEST Set).}")
    print(r"\label{tab:exact_match}")
    print(r"\begin{tabular}{l l c c}")
    print(r"\toprule")
    print(r"\textbf{Outcome Type} & \textbf{Model} & \textbf{EM (Zero-Shot)} & \textbf{EM (Few-Shot)} \\")
    print(r"\midrule")

    for display_name, json_key in subsets:
        print(f"\\multirow{{{len(available_models)}}}{{*}}{{\\textbf{{{display_name}}}}}")
        
        for model in available_models:
            row_str = f" & {model} "
            
            for setting in settings:
                data_node = results_data.get(model, {}).get(setting, {})
                
                if not data_node:
                    row_str += "& - "
                    continue

                subset_data = data_node.get(json_key, {}).get("aggregated", {})
                em_score = subset_data.get("exact_match", 0) * 100
                
                row_str += f"& {em_score:.1f}\\% "
            
            print(row_str + r"\\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

# --- Execution ---
if __name__ == "__main__":
    if os.path.exists(RESULTS_DIR):
        full_data = aggregate_results(RESULTS_DIR)
        if full_data:
            generate_latex_tables(full_data)
        else:
            print("No matching '_TEST' folders containing data found.")
    else:
        print(f"Please ensure the directory '{RESULTS_DIR}' exists.")