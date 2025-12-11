"""
Quick preview of what Evaluator._prepare_long_data does to the gold standard.
Usage (from repo root):
    python -m src.evaluation.test --split DEV --rows 5
"""
import argparse
import json

from src.config import GOLD_STANDARD_PATH
from src.evaluation.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Preview long_df for gold standard only")
    parser.add_argument("--split", default="DEV", help="Which split to filter (DEV/TEST)")
    parser.add_argument("--rows", type=int, default=5, help="Rows to display from long_df")
    args = parser.parse_args()

    with open(GOLD_STANDARD_PATH, "r", encoding="utf-8") as f:
        full_gold = json.load(f)
    gold = [row for row in full_gold if row.get("split") == args.split]

    evaluator = Evaluator(gold, [])
    long_df = evaluator.long_df

    print(f"Gold rows: {len(gold)}")
    print(f"Long DataFrame shape: {long_df.shape}")
    if long_df.empty:
        print("Long DataFrame is empty")
    else:
        print(long_df.head(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
