import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR

# Map substrings in folder names to model display names
MODEL_MAPPING: Dict[str, str] = {
    "gpt": "GPT",
    "gemini": "Gemini",
    "haiku": "Claude-Haiku",
}

# Columns that define a unique prediction target
KEY_COLS: List[str] = [
    "pmcid",
    "intervention",
    "comparator",
    "outcome",
    "outcome_type",
    "field",
]

ERROR_STATUSES = {"FP", "FN"}


def detect_model(folder_name: str) -> str:
    name = folder_name.lower()
    for key, display in MODEL_MAPPING.items():
        if key in name:
            return display
    return folder_name


def find_zero_shot_runs(results_dir: str) -> List[Tuple[str, str]]:
    runs = []
    if not os.path.exists(results_dir):
        return runs

    for entry in os.listdir(results_dir):
        full_path = os.path.join(results_dir, entry)
        if not os.path.isdir(full_path):
            continue
        lower = entry.lower()
        if "zero-shot" not in lower or "_test" not in lower:
            continue
        runs.append((detect_model(entry), full_path))
    return runs


def load_error_keys(csv_path: str) -> Set[Tuple[str, ...]]:
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    # normalize
    for col in KEY_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    cat_col = "category" if "category" in df.columns else None
    if cat_col is None:
        return set()
    df[cat_col] = df[cat_col].astype(str).str.upper()
    df["error_flag"] = df[cat_col].isin(ERROR_STATUSES)

    error_rows = df[df["error_flag"]]
    keys = set(zip(*[error_rows[c] for c in KEY_COLS]))
    return keys


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def main():
    runs = find_zero_shot_runs(str(RESULTS_DIR))
    if not runs:
        print("No zero-shot TEST runs found.")
        return

    error_sets: Dict[str, Set[Tuple[str, ...]]] = {}
    labels: List[str] = []

    for model, path in runs:
        csv_path = os.path.join(path, "evaluation_details.csv")
        label = f"{model} ({os.path.basename(path)})"
        labels.append(label)
        error_sets[label] = load_error_keys(csv_path)

    # Build similarity matrix
    n = len(labels)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                score = jaccard(error_sets[li], error_sets[lj])
                matrix[i][j] = score
                matrix[j][i] = score

    # Escape underscores for LaTeX labels
    latex_labels = [lbl.replace("_", r"\_") for lbl in labels]

    # Print LaTeX table
    print("%% LaTeX table: Pairwise Jaccard similarity on error sets (FP/FN) for zero-shot TEST runs")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Pairwise Jaccard similarity on error sets (FP/FN) for zero-shot TEST runs.}")
    print(r"\label{tab:jaccard_zero_shot_errors}")
    col_spec = "l " + " ".join(["c"] * n)
    print(r"\begin{tabular}{" + col_spec + r"}")
    print(r"\toprule")
    header = "Model & " + " & ".join(latex_labels) + r" \\"
    print(header)
    print(r"\midrule")
    for i, lbl in enumerate(latex_labels):
        row_vals = [f"{matrix[i][j]:.4f}" for j in range(n)]
        row = lbl + " & " + " & ".join(row_vals) + r" \\"
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # All-model intersection vs union (one scalar across all runs)
    all_sets = list(error_sets.values())
    if all_sets:
        inter_all = set.intersection(*all_sets) if len(all_sets) > 1 else all_sets[0]
        union_all = set.union(*all_sets) if len(all_sets) > 1 else all_sets[0]
        overall = len(inter_all) / len(union_all) if union_all else 1.0

        print("\n%% LaTeX table: Overall Jaccard (intersection/union across all zero-shot TEST runs)")
        print(r"\begin{table}[ht]")
        print(r"\centering")
        print(r"\caption{Overall Jaccard similarity on error sets (FP/FN) across all zero-shot TEST runs.}")
        print(r"\label{tab:jaccard_zero_shot_errors_overall}")
        print(r"\begin{tabular}{c c}")
        print(r"\toprule")
        print(r"\textbf{Runs} & \textbf{Jaccard} \\")
        print(r"\midrule")
        run_list = r", ".join(latex_labels)
        print(f"{run_list} & {overall:.4f} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")


if __name__ == "__main__":
    main()
