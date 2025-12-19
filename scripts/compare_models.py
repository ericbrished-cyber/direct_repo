import pandas as pd
import argparse
from pathlib import Path
import sys

def calculate_jaccard_error_similarity(file1_path, file2_path):
    print(f"Loading Model A: {file1_path}")
    df1 = pd.read_csv(file1_path)
    print(f"Loading Model B: {file2_path}")
    df2 = pd.read_csv(file2_path)

    # Identifiera unika datapunkter baserat på dina nycklar i metrics.py
    # Vi konverterar till sträng för att undvika problem om någon modell tolkat tal olika
    key_cols = ['pmcid', 'intervention', 'comparator', 'outcome', 'outcome_type', 'field']
    
    for df in [df1, df2]:
        for col in key_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")

    # Definiera "Misslyckande" (Failures) som antingen Hallucination (FP) eller Miss (FN)
    fail_status = ['FP', 'FN']
    
    failures_1 = df1[df1['category'].isin(fail_status)]
    failures_2 = df2[df2['category'].isin(fail_status)]

    print(f"\nModel A Total Failures: {len(failures_1)}")
    print(f"Model B Total Failures: {len(failures_2)}")

    # Skapa unika nycklar (tuples) för varje fel
    # Detta gör att vi kan jämföra exakt "samma" fel
    keys_1 = set(zip(*[failures_1[c] for c in key_cols if c in failures_1.columns]))
    keys_2 = set(zip(*[failures_2[c] for c in key_cols if c in failures_2.columns]))

    intersection = keys_1.intersection(keys_2)
    union = keys_1.union(keys_2)

    len_intersection = len(intersection)
    len_union = len(union)

    if len_union == 0:
        print("No failures in either model. Jaccard = 1.0 (Perfect)")
        return 1.0

    jaccard_index = len_intersection / len_union

    print("-" * 40)
    print(f"Shared Failures (Intersection): {len_intersection}")
    print(f"Total Unique Failures (Union):  {len_union}")
    print("-" * 40)
    print(f"JACCARD ERROR SIMILARITY: {jaccard_index:.4f}")
    print("-" * 40)
    
    if jaccard_index > 0.6:
        print("Interpretation: High overlap. Models find the same specific examples difficult (Systematic Error).")
    elif jaccard_index < 0.3:
        print("Interpretation: Low overlap. Models fail on different examples (Random Hallucinations/Noise).")
    else:
        print("Interpretation: Moderate overlap.")

def main():
    parser = argparse.ArgumentParser(description="Calculate Jaccard Error Similarity between two models.")
    parser.add_argument("file1", help="Path to evaluation_details.csv for Model A")
    parser.add_argument("file2", help="Path to evaluation_details.csv for Model B")
    args = parser.parse_args()
    
    f1 = Path(args.file1)
    f2 = Path(args.file2)

    if not f1.exists() or not f2.exists():
        print("Error: One or both files not found.")
        sys.exit(1)

    calculate_jaccard_error_similarity(f1, f2)

if __name__ == "__main__":
    main()