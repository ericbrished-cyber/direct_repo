import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Any, Dict, List, Tuple

from evaluation import evaluate, NUMERIC_FIELDS, GOLD_PATH_DEFAULT
from utils import get_pmcid_from_filename


class BatchEvaluator:
    """
    Handles batch evaluation of multiple articles and aggregates results.
    """

    def __init__(self, gold_path: str = GOLD_PATH_DEFAULT, output_dir: str = "evaluation_results"):
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

        for i, pred_file in enumerate(sorted(pred_files), 1):
            try:
                pmcid = get_pmcid_from_filename(str(pred_file))

                # Check if PMCID exists in gold standard
                if pmcid not in self.gold_pmcids:
                    print(f"[{i}/{len(pred_files)}] PMCID={pmcid} ⚠ not in gold standard, skipping")
                    continue

                pred_data = self._load_json(pred_file)
                gold_fields = self._count_gold_fields(pmcid)
                pred_fields = self._count_pred_fields(pred_data)

                metrics = evaluate(self.gold_path, str(pred_file), verbose=False)

                # Store results
                result = {
                    "pmcid": pmcid,
                    "filename": pred_file.name,
                    "gold_fields": gold_fields,
                    "pred_fields": pred_fields,
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "per_field_metrics": metrics.get("per_field_metrics", {}),
                }
                results_list.append(result)

                print(
                    f"[{i}/{len(pred_files)}] PMCID={pmcid} - "
                    f"P: {metrics['precision']:.3f}, "
                    f"R: {metrics['recall']:.3f}, "
                    f"F1: {metrics['f1']:.3f}"
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
                "per_field_micro": {field: {} for field in NUMERIC_FIELDS},
                "total_gold_fields": 0,
                "total_pred_fields": 0,
                "total_tp": 0,
                "total_fp": 0,
                "total_fn": 0,
                "total_tn": 0,
            }

        # Macro averages (average of per-article metrics)
        macro_precision = sum(r["precision"] for r in results_list) / len(results_list)
        macro_recall = sum(r["recall"] for r in results_list) / len(results_list)
        macro_f1 = sum(r["f1"] for r in results_list) / len(results_list)

        # Micro averages (aggregate TP/FP/FN across all articles)
        total_tp = sum(r["tp"] for r in results_list)
        total_fp = sum(r["fp"] for r in results_list)
        total_fn = sum(r["fn"] for r in results_list)
        total_tn = sum(r["tn"] for r in results_list)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall)
            else 0.0
        )

        # Per-field micro metrics across all articles
        per_field_totals: Dict[str, Dict[str, int]] = {
            field: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for field in NUMERIC_FIELDS
        }
        for r in results_list:
            pfm = r.get("per_field_metrics", {})
            for field in NUMERIC_FIELDS:
                counts = pfm.get(field, {})
                per_field_totals[field]["tp"] += counts.get("tp", 0)
                per_field_totals[field]["fp"] += counts.get("fp", 0)
                per_field_totals[field]["fn"] += counts.get("fn", 0)
                per_field_totals[field]["tn"] += counts.get("tn", 0)

        per_field_micro: Dict[str, Dict[str, float]] = {}
        for field, counts in per_field_totals.items():
            tp_f = counts["tp"]
            fp_f = counts["fp"]
            fn_f = counts["fn"]
            precision_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else 0.0
            recall_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) else 0.0
            f1_f = (
                2 * precision_f * recall_f / (precision_f + recall_f)
                if (precision_f + recall_f)
                else 0.0
            )
            per_field_micro[field] = {
                "tp": tp_f,
                "fp": fp_f,
                "fn": fn_f,
                "tn": counts["tn"],
                "precision": precision_f,
                "recall": recall_f,
                "f1": f1_f,
            }

        return {
            "run_name": run_name,
            "num_articles": len(results_list),
            "total_gold_fields": sum(r["gold_fields"] for r in results_list),
            "total_pred_fields": sum(r["pred_fields"] for r in results_list),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
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
            f.write(f"Total gold fields: {aggregated['total_gold_fields']}\n")
            f.write(f"Total predicted fields: {aggregated['total_pred_fields']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MICRO-AVERAGED METRICS (aggregate TP / FP / FN)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  TP:           {aggregated['total_tp']}\n")
            f.write(f"  FP:           {aggregated['total_fp']}\n")
            f.write(f"  FN:           {aggregated['total_fn']}\n")
            f.write(f"  TN:           {aggregated['total_tn']}\n")
            f.write(f"  Precision:    {aggregated['micro_precision']:.4f}\n")
            f.write(f"  Recall:       {aggregated['micro_recall']:.4f}\n")
            f.write(f"  F1:           {aggregated['micro_f1']:.4f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MACRO-AVERAGED METRICS (average of per-article metrics)\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Precision: {aggregated['macro_precision']:.4f}\n")
            f.write(f"  Recall:    {aggregated['macro_recall']:.4f}\n")
            f.write(f"  F1:        {aggregated['macro_f1']:.4f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("PER-FIELD MICRO METRICS (aggregate TP / FP / FN / TN)\n")
            f.write("-" * 80 + "\n")
            for field in NUMERIC_FIELDS:
                m = aggregated["per_field_micro"].get(field, {})
                f.write(
                    f"{field:30s} "
                    f"P: {m.get('precision', 0):.4f}  "
                    f"R: {m.get('recall', 0):.4f}  "
                    f"F1: {m.get('f1', 0):.4f}  "
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
                        f"  TP / FP / FN / TN: "
                        f"{r['tp']} / {r['fp']} / {r['fn']} / {r['tn']}\n"
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

    def _load_json(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]

    def _count_gold_fields(self, pmcid: int) -> int:
        return sum(
            1
            for row in self.gold_data
            if int(row.get("pmcid")) == int(pmcid)
            for field in NUMERIC_FIELDS
            if row.get(field) is not None
        )

    def _count_pred_fields(self, pred_data: Any) -> int:
        def _rows(obj: Any):
            if isinstance(obj, dict):
                return obj.get("rows") or obj.get("extractions") or []
            return []

        rows: List[Dict[str, Any]] = []
        if isinstance(pred_data, list):
            for obj in pred_data:
                rows.extend(_rows(obj))
        else:
            rows = _rows(pred_data)

        count = 0
        for row in rows:
            for field in NUMERIC_FIELDS:
                if row.get(field) is not None:
                    count += 1
        return count

def main():
    """
    Example usage of batch evaluation.
    """
    # Initialize evaluator
    evaluator = BatchEvaluator(
        gold_path="gold-standard/gold_standard_clean.json",
        output_dir="evaluation_results",
    )

    # Example 1: Evaluate guided PDF extractions
    evaluator.evaluate_directory(
        predictions_dir="outputs/gpt_5_1_fewshot_20251208_125704/extractions",
        suffix_filter="",
        run_name="test",
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
