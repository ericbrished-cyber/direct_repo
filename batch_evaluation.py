import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Any, Dict, List

from evaluation import evaluate, NUMERIC_FIELDS, GOLD_PATH_DEFAULT
from utils import get_pmcid_from_filename


class BatchEvaluator:
    """
    Handles batch evaluation of multiple articles and aggregates results.
    
    Computes and aggregates:
    - Precision, Recall, F1 (macro and micro averages)
    - MSE and RMSE (mean squared error metrics)
    - Exact match accuracy (ICO-level correctness)
    """

    def __init__(self, gold_path: str = GOLD_PATH_DEFAULT, output_dir: str = "evaluation_results"):
        """
        Initialize the batch evaluator.
        
        Args:
            gold_path: Path to gold standard JSON file
            output_dir: Directory to save evaluation results
        """
        self.gold_path = gold_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load gold standard once for efficiency
        with open(gold_path, "r", encoding="utf-8") as f:
            self.gold_data = json.load(f)

        # Extract unique PMCIDs from gold standard
        self.gold_pmcids = set(int(row["pmcid"]) for row in self.gold_data)

    def evaluate_directory(
        self,
        predictions_dir: str,
        suffix_filter: str = None,
        run_name: str = None,
    ) -> Dict:
        """
        Evaluate all prediction files in a directory.
        
        Process:
        1. Find all prediction files matching the suffix filter
        2. Evaluate each file against gold standard
        3. Aggregate metrics across all files
        4. Save detailed results and summary
        
        Args:
            predictions_dir: Directory containing JSONL prediction files
            suffix_filter: Optional suffix to filter files (e.g., "_guided_pdf")
            run_name: Name for this evaluation run (for results file)

        Returns:
            Dictionary with aggregated results including all metrics
        """
        # Generate run name if not provided
        if run_name is None:
            run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Find prediction files (.json or .jsonl)
        pattern = f"*{suffix_filter}*.json*" if suffix_filter else "*.json*"
        pred_files = sorted(
            [p for p in Path(predictions_dir).glob(pattern) if p.suffix in {".json", ".jsonl"}]
        )

        if not pred_files:
            raise ValueError(f"No prediction files found in {predictions_dir} with pattern {pattern}")

        print(f"Found {len(pred_files)} prediction files to evaluate")
        print(f"Run name: {run_name}")
        print("=" * 80)

        # Store per-file results
        results_list: List[Dict] = []

        # Evaluate each prediction file
        for i, pred_file in enumerate(sorted(pred_files), 1):
            try:
                # Extract PMCID from filename
                pmcid = get_pmcid_from_filename(str(pred_file))

                # Check if PMCID exists in gold standard
                if pmcid not in self.gold_pmcids:
                    print(f"[{i}/{len(pred_files)}] PMCID={pmcid} ⚠ not in gold standard, skipping")
                    continue

                # Load prediction data for field counting
                pred_data = self._load_json(pred_file)
                gold_fields = self._count_gold_fields(pmcid)
                pred_fields = self._count_pred_fields(pred_data)

                # Run comprehensive evaluation
                metrics = evaluate(self.gold_path, str(pred_file), verbose=False)

                # Store all metrics for this file
                result = {
                    # Basic info
                    "pmcid": pmcid,
                    "filename": pred_file.name,
                    "gold_fields": gold_fields,
                    "pred_fields": pred_fields,
                    
                    # Confusion matrix
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    
                    # Classification metrics
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    
                    # Error metrics
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                    "num_mse_comparisons": metrics["num_mse_comparisons"],
                    
                    # Exact match metrics
                    "exact_match_accuracy": metrics["exact_match_accuracy"],
                    "exact_match_correct": metrics["exact_match_correct"],
                    "exact_match_total": metrics["exact_match_total"],
                    "exact_match_by_type": metrics["exact_match_by_type"],
                    
                    # Per-field metrics
                    "per_field_metrics": metrics.get("per_field_metrics", {}),
                }
                results_list.append(result)

                # Print progress with key metrics
                print(
                    f"[{i}/{len(pred_files)}] PMCID={pmcid} - "
                    f"P: {metrics['precision']:.3f}, "
                    f"R: {metrics['recall']:.3f}, "
                    f"F1: {metrics['f1']:.3f}, "
                    f"MSE: {metrics['mse']:.3f}, "
                    f"ExactMatch: {metrics['exact_match_accuracy']:.3f}"
                )

            except Exception as e:
                print(f"[{i}/{len(pred_files)}] {pred_file.name} ✗ error: {e}")
                continue

        # Aggregate results across all files
        aggregated = self._aggregate_results(results_list, run_name)

        # Save all results to disk
        self._save_results(results_list, aggregated, run_name, suffix_filter)

        return aggregated

    def _aggregate_results(self, results_list: List[Dict], run_name: str) -> Dict:
        """
        Compute macro and micro averages from individual results.
        
        Aggregation strategies:
        - Macro averages: Mean of per-article metrics (each article weighted equally)
        - Micro averages: Aggregate TP/FP/FN across all articles (each field weighted equally)
        - MSE: Mean of all squared errors across articles
        - Exact match: Total correct / total attempted
        
        Args:
            results_list: List of per-article result dictionaries
            run_name: Name of this evaluation run
            
        Returns:
            Dictionary with all aggregated metrics
        """
        # Handle empty results
        if not results_list:
            return {
                "run_name": run_name,
                "num_articles": 0,
                
                # Macro metrics (all zero)
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                
                # Micro metrics (all zero)
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                
                # Error metrics (all zero)
                "overall_mse": 0.0,
                "overall_rmse": 0.0,
                "total_mse_comparisons": 0,
                
                # Exact match metrics (all zero)
                "exact_match_accuracy": 0.0,
                "exact_match_correct": 0,
                "exact_match_total": 0,
                
                # Confusion matrix totals
                "total_gold_fields": 0,
                "total_pred_fields": 0,
                "total_tp": 0,
                "total_fp": 0,
                "total_fn": 0,
                "total_tn": 0,
                
                # Per-field metrics
                "per_field_micro": {field: {} for field in NUMERIC_FIELDS},
            }

        # ===== MACRO AVERAGES =====
        # Average of per-article metrics (treats each article equally)
        macro_precision = sum(r["precision"] for r in results_list) / len(results_list)
        macro_recall = sum(r["recall"] for r in results_list) / len(results_list)
        macro_f1 = sum(r["f1"] for r in results_list) / len(results_list)

        # ===== MICRO AVERAGES =====
        # Aggregate TP/FP/FN across all articles, then calculate metrics
        # This treats each field equally regardless of which article it's from
        total_tp = sum(r["tp"] for r in results_list)
        total_fp = sum(r["fp"] for r in results_list)
        total_fn = sum(r["fn"] for r in results_list)
        total_tn = sum(r["tn"] for r in results_list)

        # Calculate micro-averaged precision/recall/F1
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall)
            else 0.0
        )

        # ===== ERROR METRICS =====
        # Aggregate MSE: weighted average across all articles based on number of comparisons
        total_mse_sum = sum(
            r["mse"] * r["num_mse_comparisons"]
            for r in results_list
            if r["num_mse_comparisons"] > 0
        )
        total_mse_comparisons = sum(r["num_mse_comparisons"] for r in results_list)
        overall_mse = total_mse_sum / total_mse_comparisons if total_mse_comparisons > 0 else 0.0
        overall_rmse = (overall_mse ** 0.5) if overall_mse > 0 else 0.0

        # ===== EXACT MATCH METRICS =====
        # Total correct exact matches across all articles
        total_exact_match_correct = sum(r["exact_match_correct"] for r in results_list)
        total_exact_match_total = sum(r["exact_match_total"] for r in results_list)
        exact_match_accuracy = (
            total_exact_match_correct / total_exact_match_total
            if total_exact_match_total > 0
            else 0.0
        )
        
        # Aggregate by outcome type (binary vs continuous)
        binary_correct = sum(
            r["exact_match_by_type"]["binary"]["correct"]
            for r in results_list
        )
        binary_total = sum(
            r["exact_match_by_type"]["binary"]["total"]
            for r in results_list
        )
        continuous_correct = sum(
            r["exact_match_by_type"]["continuous"]["correct"]
            for r in results_list
        )
        continuous_total = sum(
            r["exact_match_by_type"]["continuous"]["total"]
            for r in results_list
        )
        
        exact_match_by_type = {
            "binary": {
                "correct": binary_correct,
                "total": binary_total,
                "accuracy": binary_correct / binary_total if binary_total > 0 else 0.0,
            },
            "continuous": {
                "correct": continuous_correct,
                "total": continuous_total,
                "accuracy": continuous_correct / continuous_total if continuous_total > 0 else 0.0,
            },
        }

        # ===== PER-FIELD MICRO METRICS =====
        # Aggregate TP/FP/FN/TN for each field across all articles
        per_field_totals: Dict[str, Dict[str, int]] = {
            field: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "mse_sum": 0.0, "mse_count": 0}
            for field in NUMERIC_FIELDS
        }
        
        # Sum up counts for each field
        for r in results_list:
            pfm = r.get("per_field_metrics", {})
            for field in NUMERIC_FIELDS:
                counts = pfm.get(field, {})
                per_field_totals[field]["tp"] += counts.get("tp", 0)
                per_field_totals[field]["fp"] += counts.get("fp", 0)
                per_field_totals[field]["fn"] += counts.get("fn", 0)
                per_field_totals[field]["tn"] += counts.get("tn", 0)
                
                # Aggregate MSE for this field
                if counts.get("num_comparisons", 0) > 0:
                    per_field_totals[field]["mse_sum"] += (
                        counts.get("mse", 0) * counts.get("num_comparisons", 0)
                    )
                    per_field_totals[field]["mse_count"] += counts.get("num_comparisons", 0)

        # Calculate per-field metrics from aggregated counts
        per_field_micro: Dict[str, Dict[str, float]] = {}
        for field, counts in per_field_totals.items():
            tp_f = counts["tp"]
            fp_f = counts["fp"]
            fn_f = counts["fn"]
            
            # Precision, recall, F1 for this field
            precision_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 0.0
            recall_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) else 0.0
            f1_f = (
                2 * precision_f * recall_f / (precision_f + recall_f)
                if (precision_f + recall_f)
                else 0.0
            )
            
            # MSE for this field
            mse_f = (
                counts["mse_sum"] / counts["mse_count"]
                if counts["mse_count"] > 0
                else 0.0
            )
            rmse_f = (mse_f ** 0.5) if mse_f > 0 else 0.0
            
            per_field_micro[field] = {
                "tp": tp_f,
                "fp": fp_f,
                "fn": fn_f,
                "tn": counts["tn"],
                "precision": precision_f,
                "recall": recall_f,
                "f1": f1_f,
                "mse": mse_f,
                "rmse": rmse_f,
                "num_comparisons": counts["mse_count"],
            }

        # Return comprehensive aggregated results
        return {
            # Run metadata
            "run_name": run_name,
            "num_articles": len(results_list),
            
            # Summary counts
            "total_gold_fields": sum(r["gold_fields"] for r in results_list),
            "total_pred_fields": sum(r["pred_fields"] for r in results_list),
            
            # Confusion matrix totals
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn,
            
            # Macro-averaged metrics (mean of per-article metrics)
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            
            # Micro-averaged metrics (aggregated TP/FP/FN)
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            
            # Error metrics
            "overall_mse": overall_mse,
            "overall_rmse": overall_rmse,
            "total_mse_comparisons": total_mse_comparisons,
            
            # Exact match metrics
            "exact_match_accuracy": exact_match_accuracy,
            "exact_match_correct": total_exact_match_correct,
            "exact_match_total": total_exact_match_total,
            "exact_match_by_type": exact_match_by_type,
            
            # Per-field micro-averaged metrics
            "per_field_micro": per_field_micro,
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
        
        Creates three output files:
        1. CSV with per-article results (for detailed analysis)
        2. JSON with aggregated metrics (for programmatic access)
        3. Text summary report (for human reading)
        
        Args:
            results_list: List of per-article result dictionaries
            aggregated: Dictionary of aggregated metrics
            run_name: Name of this evaluation run
            suffix_filter: Optional suffix that was used to filter files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{run_name}_{timestamp}"

        # 1. Save per-article results as CSV
        if results_list:
            # Flatten results for CSV (exclude nested dicts)
            flat_results = []
            for r in results_list:
                flat_r = {k: v for k, v in r.items() if k not in ["per_field_metrics", "exact_match_by_type"]}
                # Add exact match by type as separate columns
                flat_r["exact_match_binary_correct"] = r["exact_match_by_type"]["binary"]["correct"]
                flat_r["exact_match_binary_total"] = r["exact_match_by_type"]["binary"]["total"]
                flat_r["exact_match_continuous_correct"] = r["exact_match_by_type"]["continuous"]["correct"]
                flat_r["exact_match_continuous_total"] = r["exact_match_by_type"]["continuous"]["total"]
                flat_results.append(flat_r)
            
            df = pd.DataFrame(flat_results)
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
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION SUMMARY: {run_name}\n")
            f.write("=" * 80 + "\n\n")

            # Basic info
            if suffix_filter:
                f.write(f"Filter: {suffix_filter}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Articles evaluated: {aggregated['num_articles']}\n")
            f.write(f"Total gold fields: {aggregated['total_gold_fields']}\n")
            f.write(f"Total predicted fields: {aggregated['total_pred_fields']}\n\n")

            # Micro-averaged metrics (most important for overall performance)
            f.write("-" * 80 + "\n")
            f.write("MICRO-AVERAGED METRICS (aggregate TP/FP/FN across all articles)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  TP:           {aggregated['total_tp']}\n")
            f.write(f"  FP:           {aggregated['total_fp']}\n")
            f.write(f"  FN:           {aggregated['total_fn']}\n")
            f.write(f"  TN:           {aggregated['total_tn']}\n")
            f.write(f"  Precision:    {aggregated['micro_precision']:.4f}\n")
            f.write(f"  Recall:       {aggregated['micro_recall']:.4f}\n")
            f.write(f"  F1:           {aggregated['micro_f1']:.4f}\n\n")

            # Macro-averaged metrics
            f.write("-" * 80 + "\n")
            f.write("MACRO-AVERAGED METRICS (average of per-article metrics)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Precision: {aggregated['macro_precision']:.4f}\n")
            f.write(f"  Recall:    {aggregated['macro_recall']:.4f}\n")
            f.write(f"  F1:        {aggregated['macro_f1']:.4f}\n\n")

            # Error metrics
            f.write("-" * 80 + "\n")
            f.write("ERROR METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  MSE (Mean Squared Error):       {aggregated['overall_mse']:.4f}\n")
            f.write(f"  RMSE (Root Mean Squared Error): {aggregated['overall_rmse']:.4f}\n")
            f.write(f"  Number of comparisons:          {aggregated['total_mse_comparisons']}\n\n")

            # Exact match metrics
            f.write("-" * 80 + "\n")
            f.write("EXACT MATCH ACCURACY (all fields correct for an ICO)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Overall:    {aggregated['exact_match_accuracy']:.4f} "
                   f"({aggregated['exact_match_correct']}/{aggregated['exact_match_total']})\n")
            f.write(f"  Binary:     {aggregated['exact_match_by_type']['binary']['accuracy']:.4f} "
                   f"({aggregated['exact_match_by_type']['binary']['correct']}/"
                   f"{aggregated['exact_match_by_type']['binary']['total']})\n")
            f.write(f"  Continuous: {aggregated['exact_match_by_type']['continuous']['accuracy']:.4f} "
                   f"({aggregated['exact_match_by_type']['continuous']['correct']}/"
                   f"{aggregated['exact_match_by_type']['continuous']['total']})\n\n")

            # Per-field micro metrics
            f.write("-" * 80 + "\n")
            f.write("PER-FIELD MICRO METRICS (aggregate TP/FP/FN/TN for each field)\n")
            f.write("-" * 80 + "\n")
            for field in NUMERIC_FIELDS:
                m = aggregated["per_field_micro"].get(field, {})
                f.write(
                    f"{field:40s} "
                    f"P: {m.get('precision', 0):.4f}  "
                    f"R: {m.get('recall', 0):.4f}  "
                    f"F1: {m.get('f1', 0):.4f}  "
                    f"MSE: {m.get('mse', 0):.4f}  "
                    f"(TP={m.get('tp', 0)}, FP={m.get('fp', 0)}, "
                    f"FN={m.get('fn', 0)}, TN={m.get('tn', 0)})\n"
                )
            f.write("\n")

            # Per-article breakdown
            if results_list:
                f.write("-" * 80 + "\n")
                f.write("PER-ARTICLE BREAKDOWN\n")
                f.write("-" * 80 + "\n")
                for r in sorted(results_list, key=lambda x: x["pmcid"]):
                    f.write(f"\nPMCID={r['pmcid']} ({r['filename']}):\n")
                    f.write(f"  Gold/Pred fields: {r['gold_fields']}/{r['pred_fields']}\n")
                    f.write(
                        f"  TP/FP/FN/TN: "
                        f"{r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}\n"
                    )
                    f.write(
                        f"  P/R/F1: {r['precision']:.3f} / "
                        f"{r['recall']:.3f} / {r['f1']:.3f}\n"
                    )
                    f.write(f"  MSE: {r['mse']:.3f} ({r['num_mse_comparisons']} comparisons)\n")
                    f.write(
                        f"  Exact match: {r['exact_match_accuracy']:.3f} "
                        f"({r['exact_match_correct']}/{r['exact_match_total']})\n"
                    )

        print(f"✓ Summary report saved to: {report_path}\n")

        # Print summary to console
        print("=" * 80)
        print(f"FINAL RESULTS: {run_name}")
        print("=" * 80)
        print(f"Articles: {aggregated['num_articles']}")
        
        print(f"\nMicro-averaged (field-level):")
        print(f"  Precision: {aggregated['micro_precision']:.4f}")
        print(f"  Recall:    {aggregated['micro_recall']:.4f}")
        print(f"  F1:        {aggregated['micro_f1']:.4f}")
        
        print(f"\nMacro-averaged (article-level):")
        print(f"  Precision: {aggregated['macro_precision']:.4f}")
        print(f"  Recall:    {aggregated['macro_recall']:.4f}")
        print(f"  F1:        {aggregated['macro_f1']:.4f}")
        
        print(f"\nError metrics:")
        print(f"  MSE:  {aggregated['overall_mse']:.4f}")
        print(f"  RMSE: {aggregated['overall_rmse']:.4f}")
        
        print(f"\nExact match accuracy:")
        print(f"  Overall:    {aggregated['exact_match_accuracy']:.4f}")
        print(f"  Binary:     {aggregated['exact_match_by_type']['binary']['accuracy']:.4f}")
        print(f"  Continuous: {aggregated['exact_match_by_type']['continuous']['accuracy']:.4f}")
        
        print("=" * 80)

    def _load_json(self, path: Path) -> Any:
        """
        Load JSON or JSONL file.
        
        Handles both standard JSON and newline-delimited JSON (JSONL).
        
        Args:
            path: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(path, "r", encoding="utf-8") as f:
            try:
                # Try standard JSON first
                return json.load(f)
            except json.JSONDecodeError:
                # Fall back to JSONL (one JSON object per line)
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]

    def _count_gold_fields(self, pmcid: int) -> int:
        """
        Count total non-null numeric fields in gold standard for a PMCID.
        
        Args:
            pmcid: Article PMCID
            
        Returns:
            Count of non-null numeric fields
        """
        return sum(
            1
            for row in self.gold_data
            if int(row.get("pmcid")) == int(pmcid)
            for field in NUMERIC_FIELDS
            if row.get(field) is not None
        )

    def _count_pred_fields(self, pred_data: Any) -> int:
        """
        Count total non-null numeric fields in predictions.
        
        Handles various prediction formats.
        
        Args:
            pred_data: Prediction data (dict or list)
            
        Returns:
            Count of non-null numeric fields
        """
        def _rows(obj: Any):
            """Extract rows from a prediction object."""
            if isinstance(obj, dict):
                return obj.get("rows") or obj.get("extractions") or []
            return []

        # Extract all rows
        rows: List[Dict[str, Any]] = []
        if isinstance(pred_data, list):
            for obj in pred_data:
                rows.extend(_rows(obj))
        else:
            rows = _rows(pred_data)

        # Count non-null fields
        count = 0
        for row in rows:
            for field in NUMERIC_FIELDS:
                if row.get(field) is not None:
                    count += 1
        return count


def main():
    """
    Example usage of batch evaluation with all metrics.
    """
    # Initialize evaluator
    evaluator = BatchEvaluator(
        gold_path="gold-standard/gold_standard_clean.json",
        output_dir="evaluation_results",
    )

    # Evaluate a directory of predictions
    evaluator.evaluate_directory(
        predictions_dir="outputs/gpt_5_1_fewshot_20251208_125704/extractions",
        suffix_filter="",
        run_name="test",
    )


if __name__ == "__main__":
    main()