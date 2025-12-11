import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error

class Evaluator:
    def __init__(self, gold_standard: List[Dict], extractions: List[Dict]):
        self.gold_df = pd.DataFrame(gold_standard)
        self.extractions_df = pd.DataFrame(extractions)
        
        # Keys for aligning rows
        self.id_cols = ['intervention', 'comparator', 'outcome', 'outcome_type']
        
        # Prepare the long-format data
        self.long_df = self._prepare_long_data()

    def _prepare_long_data(self):
        """
        Transforms data from Wide to Long format and classifies each item.
        """
        numeric_fields = [
            'intervention_group_size', 'comparator_group_size',
            'intervention_mean', 'intervention_standard_deviation',
            'comparator_mean', 'comparator_standard_deviation',
            'intervention_events', 'comparator_events'
        ]
        
        if self.gold_df.empty:
            return pd.DataFrame(columns=self.id_cols + ['field', 'gold', 'pred', 'category'])
        
        # Melt Gold
        gold_id_vars = [c for c in self.id_cols if c in self.gold_df.columns]
        g_melt = self.gold_df.melt(
            id_vars=gold_id_vars, 
            value_vars=[f for f in numeric_fields if f in self.gold_df.columns], 
            var_name='field', value_name='gold'
        )
        
        # Melt Extractions
        if self.extractions_df.empty:
            m_melt = pd.DataFrame(columns=self.id_cols + ['field', 'pred'])
        else:
            ext_id_vars = [c for c in self.id_cols if c in self.extractions_df.columns]
            m_melt = self.extractions_df.melt(
                id_vars=ext_id_vars, 
                value_vars=[f for f in numeric_fields if f in self.extractions_df.columns], 
                var_name='field', value_name='pred'
            )
        
        # Merge
        merge_keys = self.id_cols + ['field']
        merged = pd.merge(g_melt, m_melt, on=merge_keys, how='outer')
        
        # Classify
        merged['category'] = merged.apply(self._get_row_category, axis=1)
        return merged

    def _get_row_category(self, row):
        gold = row['gold']
        pred = row['pred']
        gold_exists = pd.notna(gold)
        pred_exists = pd.notna(pred)
        
        if gold_exists:
            if pred_exists and self._is_match(gold, pred):
                return 'TP'
            else:
                return 'FN' # Missed or Wrong Value
        else:
            if pred_exists:
                return 'FP' # Hallucination
            else:
                return 'TN'

    def _is_match(self, val1, val2, tolerance=1e-3):
        try:
            return np.isclose(float(val1), float(val2), atol=tolerance)
        except (ValueError, TypeError):
            return False

    def _compute_stats(self, df_subset):
        """Helper to compute P/R/F1/RMSE on a specific dataframe slice."""
        if df_subset.empty:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "rmse": 0.0, "tp":0, "fp":0, "fn":0}

        counts = df_subset['category'].value_counts()
        TP = counts.get('TP', 0)
        FP = counts.get('FP', 0)
        FN = counts.get('FN', 0)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # RMSE only on intersections
        rmse_subset = df_subset[pd.notna(df_subset['gold']) & pd.notna(df_subset['pred'])]
        if len(rmse_subset) > 0:
            rmse = np.sqrt(mean_squared_error(rmse_subset['gold'], rmse_subset['pred']))
        else:
            rmse = 0.0
            
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rmse": rmse,
            "true_positives": int(TP),
            "false_positives": int(FP),
            "false_negatives": int(FN)
        }

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Main entry point. Returns:
        {
            "aggregated": { ... },
            "by_field": { "intervention_mean": {...}, ... }
        }
        """
        # 1. Aggregated Metrics (All fields combined)
        agg_stats = self._compute_stats(self.long_df)
        
        # 2. Exact Match (ICO level)
        exact_matches = []
        if not self.long_df.empty:
            groups = self.long_df.groupby(self.id_cols)
            for _, group in groups:
                is_perfect = group['category'].isin(['TP', 'TN']).all()
                exact_matches.append(1 if is_perfect else 0)
        
        agg_stats['exact_match'] = np.mean(exact_matches) if exact_matches else 0.0

        # 3. Per-Field Metrics
        by_field = {}
        if not self.long_df.empty:
            for field_name, group in self.long_df.groupby('field'):
                by_field[field_name] = self._compute_stats(group)

        return {
            "aggregated": agg_stats,
            "by_field": by_field
        }

def calculate_metrics(extractions: List[Dict], gold_standard: List[Dict]) -> Dict[str, Any]:
    evaluator = Evaluator(gold_standard, extractions)
    return evaluator.calculate_metrics()