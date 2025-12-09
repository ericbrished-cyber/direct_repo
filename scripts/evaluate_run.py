import json
import argparse
import sys
import os
from pathlib import Path

# Se till att Python hittar 'src' oavsett hur du kör scriptet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GOLD_STANDARD_PATH, RESULTS_DIR
from src.evaluation.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate precision/recall for a run.")
    parser.add_argument("--run_folder", type=str, required=True, help="Name of the run folder in data/results/")
    parser.add_argument("--split", type=str, default="DEV", help="Split to evaluate against (DEV or TEST)")
    args = parser.parse_args()

    # 1. Ladda Gold Standard (Facit)
    print(f"Loading Gold Standard from: {GOLD_STANDARD_PATH}")
    try:
        with open(GOLD_STANDARD_PATH, 'r', encoding='utf-8') as f:
            full_gold_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find gold standard at {GOLD_STANDARD_PATH}")
        return

    # Filtrera facit
    gold_standard = [item for item in full_gold_data if item.get("split") == args.split]
    print(f"Found {len(gold_standard)} ground truth items for split '{args.split}'")

    # 2. Ladda dina resultat
    results_path = RESULTS_DIR / args.run_folder / "final_results.json"
    if not results_path.exists():
        print(f"Error: Could not find {results_path}")
        return

    print(f"Loading predictions from: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    print(f"Found {len(predictions)} extracted items.")

    # 3. Kör beräkningen
    if not gold_standard:
        print("Warning: No gold standard items found to compare against!")
        return

    metrics = calculate_metrics(predictions, gold_standard)

    # 4. Visa resultatet
    print("\n" + "="*30)
    print(f"RESULTS FOR {args.run_folder}")
    print("="*30)
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1']:.2%}")
    print("-" * 30)
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print("="*30)

if __name__ == "__main__":
    main() 