import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple

from evaluation import (
    build_gold_arm_facts,
    build_pred_arm_facts,
    evaluate_arm_facts_open_world,  # NEW
    get_pmcid_from_filename,
)


class BatchEvaluator:
    """
    Handles batch evaluation of multiple articles and aggregates results.
    """

    def __init__(self, gold_path: str, output_dir: str = "evaluation_results"):
        """
        Args:
            gold_path: Path to gold standard JSON file
            output_dir: Directory to save evaluation results
        """
        self.gold_path = gold_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load gold standard once
        with open(gold_path, "r", encoding="utf-8") as f:
            self.gold_data = json.load(f)

        # Get unique PMCIDs from gold standard
        self.gold_pmcids = set(int(row["pmcid"]) for row in self.gold_data)

    def evaluate_directory(
        self,
        predictions_dir: str,
        suffix_filter: str = None,
        run_name: str = None,
    ) -> Dict:
        """
        Evaluate all prediction files in a directory.

        Args:
            predictions_dir: Directory containing JSONL prediction files
            suffix_filter: Optional suffix to filter files (e.g., "_guided_pdf")
            run_name: Name for this evaluation run (for results file)

        Returns:
            Dictionary with aggregated results
        """
        if run_name is None:
            run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Find all JSONL files
        pattern = f"*{suffix_filter}.jsonl" if suffix_filter else "*.jsonl"
        pred_files = list(Path(predictions_dir).glob(pattern))

        if not pred_files:
            raise ValueError(f"No prediction files found in {predictions_dir} with pattern {pattern}")

        print(f"Found {len(pred_files)} prediction files to evaluate")
        print(f"Run name: {run_name}")
        print("=" * 80)

        # Store per-file results
        results_list: List[Dict] = []

        for i, pred_file in enumerate(sorted(pred_files), 1):
            try:
                pmcid = get_pmcid_from_filename(str(pred_file))

                # Check if PMCID exists in gold standard
                if pmcid not in self.gold_pmcids:
                    print(f"[{i}/{len(pred_files)}] PMCID={pmcid} ⚠ not in gold standard, skipping")
                    continue

                # Build facts
                gold_facts = build_gold_arm_facts(self.gold_path, pmcid)
                pred_facts = build_pred_arm_facts(str(pred_file))

                # Evaluate with open-world evaluator
                metrics = evaluate_arm_facts_open_world(gold_facts, pred_facts)

                # Store results
                result = {
                    "pmcid": pmcid,
                    "filename": pred_file.name,
                    "gold_facts": len(gold_facts),
                    "pred_facts": len(pred_facts),
                    "tp": metrics["tp"],
                    "fp_in_gold": metrics["fp_in_gold"],            # NEW
                    "fn": metrics["fn"],
                    "extra_predictions": metrics["extra_predictions"],  # NEW
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
                results_list.append(result)

                print(
                    f"[{i}/{len(pred_files)}] PMCID={pmcid} - "
                    f"P: {metrics['precision']:.3f}, "
                    f"R: {metrics['recall']:.3f}, "
                    f"F1: {metrics['f1']:.3f}, "
                    f"Extra: {metrics['extra_predictions']}"
                )

            except Exception as e:
                print(f"[{i}/{len(pred_files)}] {pred_file.name} ✗ error: {e}")
                continue

        # Aggregate results
        aggregated = self._aggregate_results(results_list, run_name)

        # Save results
        self._save_results(results_list, aggregated, run_name, suffix_filter)

        return aggregated

    def _aggregate_results(self, results_list: List[Dict], run_name: str) -> Dict:
        """
        Compute macro and micro averages from individual results.

        IMPORTANT:
        - FP for precision/recall is ONLY fp_in_gold (wrong values on gold cells)
        - extra_predictions are counted and reported separately
        """
        if not results_list:
            return {
                "run_name": run_name,
                "num_articles": 0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "total_gold_facts": 0,
                "total_pred_facts": 0,
                "total_tp": 0,
                "total_fp_in_gold": 0,
                "total_fn": 0,
                "total_extra_predictions": 0,
            }

        # Macro averages (average of per-article metrics)
        macro_precision = sum(r["precision"] for r in results_list) / len(results_list)
        macro_recall = sum(r["recall"] for r in results_list) / len(results_list)
        macro_f1 = sum(r["f1"] for r in results_list) / len(results_list)

        # Micro averages (aggregate TP/FP/FN across all articles)
        total_tp = sum(r["tp"] for r in results_list)
        total_fp_in_gold = sum(r["fp_in_gold"] for r in results_list)
        total_fn = sum(r["fn"] for r in results_list)
        total_extra = sum(r["extra_predictions"] for r in results_list)

        micro_precision = total_tp / (total_tp + total_fp_in_gold) if (total_tp + total_fp_in_gold) else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall)
            else 0.0
        )

        return {
            "run_name": run_name,
            "num_articles": len(results_list),
            "total_gold_facts": sum(r["gold_facts"] for r in results_list),
            "total_pred_facts": sum(r["pred_facts"] for r in results_list),
            "total_tp": total_tp,
            "total_fp_in_gold": total_fp_in_gold,
            "total_fn": total_fn,
            "total_extra_predictions": total_extra,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }

    def _save_results(
        self,
        results_list: List[Dict],
        aggregated: Dict,
        run_name: str,
        suffix_filter: str = None,
    ):
        """
        Save evaluation results in multiple formats.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{run_name}_{timestamp}"

        # 1. Save per-article results as CSV
        if results_list:
            df = pd.DataFrame(results_list)
            csv_path = os.path.join(self.output_dir, f"{base_name}_per_article.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Per-article results saved to: {csv_path}")

        # 2. Save aggregated metrics as JSON
        json_path = os.path.join(self.output_dir, f"{base_name}_aggregated.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        print(f"✓ Aggregated metrics saved to: {json_path}")

        # 3. Save summary report as text
        report_path = os.path.join(self.output_dir, f"{base_name}_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION SUMMARY: {run_name}\n")
            f.write("=" * 80 + "\n\n")

            if suffix_filter:
                f.write(f"Filter: {suffix_filter}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Articles evaluated: {aggregated['num_articles']}\n")
            f.write(f"Total gold facts: {aggregated['total_gold_facts']}\n")
            f.write(f"Total predicted facts: {aggregated['total_pred_facts']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MICRO-AVERAGED METRICS (aggregate TP / FP_in_gold / FN)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  TP:           {aggregated['total_tp']}\n")
            f.write(f"  FP_in_gold:   {aggregated['total_fp_in_gold']}\n")
            f.write(f"  FN:           {aggregated['total_fn']}\n")
            f.write(f"  Extra preds:  {aggregated['total_extra_predictions']}\n")
            f.write(f"  Precision:    {aggregated['micro_precision']:.4f}\n")
            f.write(f"  Recall:       {aggregated['micro_recall']:.4f}\n")
            f.write(f"  F1:           {aggregated['micro_f1']:.4f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MACRO-AVERAGED METRICS (average of per-article metrics)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Precision: {aggregated['macro_precision']:.4f}\n")
            f.write(f"  Recall:    {aggregated['macro_recall']:.4f}\n")
            f.write(f"  F1:        {aggregated['macro_f1']:.4f}\n\n")

            # Per-article breakdown
            if results_list:
                f.write("-" * 80 + "\n")
                f.write("PER-ARTICLE BREAKDOWN\n")
                f.write("-" * 80 + "\n")
                for r in sorted(results_list, key=lambda x: x["pmcid"]):
                    f.write(f"\nPMCID={r['pmcid']} ({r['filename']}):\n")
                    f.write(f"  Gold/Pred facts: {r['gold_facts']}/{r['pred_facts']}\n")
                    f.write(
                        f"  TP / FP_in_gold / FN / Extra: "
                        f"{r['tp']} / {r['fp_in_gold']} / {r['fn']} / {r['extra_predictions']}\n"
                    )
                    f.write(
                        f"  P/R/F1: {r['precision']:.3f} / "
                        f"{r['recall']:.3f} / {r['f1']:.3f}\n"
                    )

        print(f"✓ Summary report saved to: {report_path}\n")

        # Print summary to console
        print("=" * 80)
        print(f"FINAL RESULTS: {run_name}")
        print("=" * 80)
        print(f"Articles: {aggregated['num_articles']}")
        print(f"\nMicro-averaged:")
        print(f"  Precision: {aggregated['micro_precision']:.4f}")
        print(f"  Recall:    {aggregated['micro_recall']:.4f}")
        print(f"  F1:        {aggregated['micro_f1']:.4f}")
        print(f"\nMacro-averaged:")
        print(f"  Precision: {aggregated['macro_precision']:.4f}")
        print(f"  Recall:    {aggregated['macro_recall']:.4f}")
        print(f"  F1:        {aggregated['macro_f1']:.4f}")
        print("=" * 80)

def main():
    """
    Example usage of batch evaluation.
    """
    # Initialize evaluator
    evaluator = BatchEvaluator(
        gold_path="gold-standard/annotated_rct_dataset.json",
        output_dir="evaluation_results",
    )

    # Example 1: Evaluate guided PDF extractions
    evaluator.evaluate_directory(
        predictions_dir="outputs",
        suffix_filter="_guided_pdf",
        run_name="guided_pdf_goldstandard_only_TP_FN",
    )

    # Example 2: Evaluate guided XML extractions
    # evaluator.evaluate_directory(
    #     predictions_dir="outputs",
    #     suffix_filter="_guided_xml",
    #     run_name="guided_xml_run1"
    # )

    # Example 3: Evaluate all extractions
    # evaluator.evaluate_directory(
    #     predictions_dir="outputs",
    #     suffix_filter="_all_xml",
    #     run_name="all_xml_run1"
    # )


if __name__ == "__main__":
    main()
