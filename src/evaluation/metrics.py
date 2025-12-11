import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import mean_squared_error

class Evaluator:
    def __init__(self, gold_standard: List[Dict], extractions: List[Dict]):
        # Convert lists of dicts to DataFrames
        self.gold_df = pd.DataFrame(gold_standard)
        self.extractions_df = pd.DataFrame(extractions)
        
        # We rely on the text strings to align the extraction to the gold standard
        self.id_cols = ['intervention', 'comparator', 'outcome', 'outcome_type']
        
        # Prepare the data
        self.long_df = self._prepare_long_data()

    def _prepare_long_data(self):
        """
        Transforms data from Wide (one row per ICO) to Long (one row per Field).
        Aligns Gold and Extraction based on the id_cols.
        """
        numeric_fields = [
            'intervention_group_size', 'comparator_group_size',
            'intervention_mean', 'intervention_standard_deviation',
            'comparator_mean', 'comparator_standard_deviation',
            'intervention_events', 'comparator_events'
        ]
        
        # 1. Handle empty dataframes gracefully
        if self.gold_df.empty:
            return pd.DataFrame(columns=self.id_cols + ['field', 'gold', 'pred', 'category'])
        
        # 2. Melt Gold Standard
        # We assume gold standard might have 'pmcid' or 'id', we keep them if present but don't match on them
        gold_id_vars = [c for c in self.id_cols if c in self.gold_df.columns]
        g_melt = self.gold_df.melt(
            id_vars=gold_id_vars, 
            value_vars=[f for f in numeric_fields if f in self.gold_df.columns], 
            var_name='field', value_name='gold'
        )
        
        # 3. Melt Extractions
        if self.extractions_df.empty:
            m_melt = pd.DataFrame(columns=self.id_cols + ['field', 'pred'])
        else:
            ext_id_vars = [c for c in self.id_cols if c in self.extractions_df.columns]
            m_melt = self.extractions_df.melt(
                id_vars=ext_id_vars, 
                value_vars=[f for f in numeric_fields if f in self.extractions_df.columns], 
                var_name='field', value_name='pred'
            )
        
        # 4. Merge strictly on the Text Columns + Field
        # This aligns "Mean" of "Intervention A" in Gold with "Mean" of "Intervention A" in Model
        merge_keys = self.id_cols + ['field']
        merged = pd.merge(g_melt, m_melt, on=merge_keys, how='outer')
        
        # 5. Classify every single field
        merged['category'] = merged.apply(self._get_row_category, axis=1)
        return merged

    def _get_row_category(self, row):
        """
        Classifies prediction based on THESIS definitions:
        - FN: Data missed OR incorrectly extracted.
        - FP: Model generated data when none existed.
        """
        gold = row['gold']
        pred = row['pred']
        
        gold_exists = pd.notna(gold)
        pred_exists = pd.notna(pred)
        
        if gold_exists:
            if pred_exists and self._is_match(gold, pred):
                return 'TP'
            else:
                # Gold exists, but model is missing OR model is wrong -> FN
                return 'FN'
        else:
            if pred_exists:
                # Gold is empty, but model predicted something -> FP
                return 'FP'
            else:
                return 'TN'

    def _is_match(self, val1, val2, tolerance=1e-3):
        try:
            # Handle float comparison
            return np.isclose(float(val1), float(val2), atol=tolerance)
        except (ValueError, TypeError):
            return False

    def calculate_metrics(self):
        """Calculates Precision, Recall, F1, RMSE, and Exact Match."""
        df = self.long_df
        if df.empty:
            return {}

        # 1. Classification Metrics (Field Level)
        counts = df['category'].value_counts()
        TP = counts.get('TP', 0)
        FP = counts.get('FP', 0)
        FN = counts.get('FN', 0)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 2. RMSE (Strictly on intersection)
        rmse_subset = df[pd.notna(df['gold']) & pd.notna(df['pred'])]
        if len(rmse_subset) > 0:
            rmse = np.sqrt(mean_squared_error(rmse_subset['gold'], rmse_subset['pred']))
        else:
            rmse = 0.0
            
        # 3. Exact Match (Aggregated by ICO)
        # Groups by the text triplets to check if ALL fields for that ICO are correct
        exact_matches = []
        # Group by the unique ICO keys
        groups = df.groupby(self.id_cols)
        
        for name, group in groups:
            # Check if all relevant fields for this specific ICO are TP or TN
            # (i.e., no errors)
            is_perfect = group['category'].isin(['TP', 'TN']).all()
            exact_matches.append(1 if is_perfect else 0)
            
        exact_match_score = np.mean(exact_matches) if exact_matches else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rmse": rmse,
            "exact_match": exact_match_score,
            "true_positives": int(TP),
            "false_positives": int(FP),
            "false_negatives": int(FN)
        }

def calculate_metrics(extractions: List[Dict], gold_standard: List[Dict]) -> Dict[str, float]:
    """
    Wrapper function to maintain compatibility with runner.py
    """
    evaluator = Evaluator(gold_standard, extractions)
    return evaluator.calculate_metrics()