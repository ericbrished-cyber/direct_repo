import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error
from difflib import SequenceMatcher

def _filter_subset(gold_standard: List[Dict], extractions: List[Dict], subset: str = None):
    """
    Optionally restrict evaluation to a subset.
    """
    if subset != "figures":
        return gold_standard, extractions

    gold_subset = [row for row in gold_standard if row.get("is_data_in_figure_graphics")]
    if not gold_subset:
        return [], []

    def _key(row: Dict[str, Any]):
        return (
            str(row.get("pmcid")),
            row.get("intervention", ""),
            row.get("comparator", ""),
            row.get("outcome", ""),
            row.get("outcome_type", ""),
        )

    allowed_keys = {_key(row) for row in gold_subset}
    pred_subset = [row for row in extractions if _key(row) in allowed_keys]
    return gold_subset, pred_subset

class Evaluator:
    def __init__(self, gold_standard: List[Dict], extractions: List[Dict]):
        # Restrict gold to only the PMCIDs present in predictions
        pmcid_filter = {str(item.get('pmcid')) for item in extractions if item.get('pmcid') is not None}
        if pmcid_filter:
            gold_standard = [row for row in gold_standard if str(row.get('pmcid')) in pmcid_filter]

        aligned_extractions = self._align_extractions(extractions, gold_standard)

        self.gold_df = pd.DataFrame(gold_standard)
        self.extractions_df = pd.DataFrame(aligned_extractions)
        
        self.id_cols = ['intervention', 'comparator', 'outcome', 'outcome_type']
        self.numeric_fields = [
            'intervention_group_size', 'comparator_group_size',
            'intervention_mean', 'intervention_standard_deviation',
            'comparator_mean', 'comparator_standard_deviation',
            'intervention_events', 'comparator_events'
        ]

        self.long_df = self._prepare_long_data()

    def _align_extractions(self, extractions: List[Dict], gold_standard: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """Aligns extraction keys to Gold Standard keys using fuzzy matching."""
        gold_map = {}
        for item in gold_standard:
            pmcid = str(item.get('pmcid'))
            if pmcid not in gold_map:
                gold_map[pmcid] = []
            
            ico_tuple = (
                item.get('intervention', ''),
                item.get('comparator', ''),
                item.get('outcome', ''),
                item.get('outcome_type', '')
            )
            gold_map[pmcid].append(ico_tuple)

        aligned_extractions = []
        for item in extractions:
            new_item = item.copy()
            pmcid = str(new_item.get('pmcid'))
            
            if pmcid in gold_map:
                candidates = gold_map[pmcid]
                query_str = f"{new_item.get('intervention', '')} {new_item.get('comparator', '')} {new_item.get('outcome', '')}"
                
                best_ratio = 0.0
                best_match = None
                
                for cand in candidates:
                    target_str = f"{cand[0]} {cand[1]} {cand[2]}"
                    ratio = SequenceMatcher(None, query_str, target_str).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = cand
                
                if best_match and best_ratio >= threshold:
                    new_item['intervention'] = best_match[0]
                    new_item['comparator'] = best_match[1]
                    new_item['outcome'] = best_match[2]
                    new_item['outcome_type'] = best_match[3]
            
            aligned_extractions.append(new_item)
        return aligned_extractions        

    def _prepare_long_data(self):
        if self.gold_df.empty:
            return pd.DataFrame(columns=self.id_cols + ['field', 'gold', 'pred', 'category'])
        
        gold_id_vars = [c for c in self.id_cols if c in self.gold_df.columns]
        g_melt = self.gold_df.melt(
            id_vars=gold_id_vars, 
            value_vars=[f for f in self.numeric_fields if f in self.gold_df.columns], 
            var_name='field', value_name='gold'
        ).assign(from_gold=True)
        
        if self.extractions_df.empty:
            m_melt = pd.DataFrame(columns=self.id_cols + ['field', 'pred', 'from_pred'])
        else:
            ext_id_vars = [c for c in self.id_cols if c in self.extractions_df.columns]
            m_melt = self.extractions_df.melt(
                id_vars=ext_id_vars, 
                value_vars=[f for f in self.numeric_fields if f in self.extractions_df.columns], 
                var_name='field', value_name='pred'
            ).assign(from_pred=True)
        
        merge_keys = self.id_cols + ['field']
        merged = pd.merge(g_melt, m_melt, on=merge_keys, how='outer')
        merged['from_gold'] = merged['from_gold'].fillna(False)
        merged['from_pred'] = merged['from_pred'].fillna(False)
        
        merged['category'] = merged.apply(self._get_row_category, axis=1)
        return merged

    def _get_row_category(self, row):
        gold = row['gold']
        pred = row['pred']
        gold_exists = pd.notna(gold)
        extraction_exists = pd.notna(pred)
        
        if gold_exists:
            if extraction_exists and self._is_match(gold, pred):
                return 'TP'
            else:
                return 'FN' 
        else:
            if extraction_exists:
                return 'FP'
            else:
                return 'TN'
        
    def _is_match(self, val1, val2, tolerance=1e-3):
        try:
            return np.isclose(float(val1), float(val2), atol=tolerance)
        except (ValueError, TypeError):
            return False

    def _compute_stats(self, df_subset):
        """Helper to compute P/R/F1/RMSE on a specific dataframe slice."""
        df_subset = df_subset[df_subset['category'] != 'IGNORE']
        if df_subset.empty:
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "rmse": 0.0,
                "true_positives": 0, "false_positives": 0, "false_negatives": 0
            }

        counts = df_subset['category'].value_counts()
        TP = counts.get('TP', 0)
        FP = counts.get('FP', 0)
        FN = counts.get('FN', 0)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        rmse_subset = df_subset[pd.notna(df_subset['gold']) & pd.notna(df_subset['pred'])]
        if len(rmse_subset) > 0:
            rmse = np.sqrt(mean_squared_error(rmse_subset['gold'], rmse_subset['pred']))
        else:
            rmse = 0.0
            
        return {
            "precision": precision, "recall": recall, "f1": f1, "rmse": rmse,
            "true_positives": int(TP), "false_positives": int(FP), "false_negatives": int(FN)
        }

    def _calculate_bootstrap_ci(self, df, metric_key, n_iterations=1000, ci=0.95):
        """
        Bootstrap resampling to calculate Confidence Intervals.
        """
        if df.empty:
            return 0.0, 0.0
            
        scores = []
        n = len(df)
        
        # Resample n_iterations times
        for _ in range(n_iterations):
            # Sample with replacement
            sample = df.sample(n=n, replace=True)
            # Re-compute stats using the exact same logic as main evaluation
            stats = self._compute_stats(sample)
            scores.append(stats[metric_key])
            
        lower = np.percentile(scores, (1 - ci) / 2 * 100)
        upper = np.percentile(scores, (1 + ci) / 2 * 100)
        return lower, upper

    def calculate_metrics(self) -> Dict[str, Any]:
        scorable_df = self.long_df[self.long_df['category'] != 'IGNORE']
        
        # 1. Aggregated Metrics
        agg_stats = self._compute_stats(scorable_df)
        
        # --- Calculate 95% Confidence Intervals ---
        if not scorable_df.empty:
            # F1 Confidence Interval
            f1_low, f1_high = self._calculate_bootstrap_ci(scorable_df, "f1")
            agg_stats["f1_ci_lower"] = f1_low
            agg_stats["f1_ci_upper"] = f1_high
            
            # RMSE Confidence Interval
            rmse_low, rmse_high = self._calculate_bootstrap_ci(scorable_df, "rmse")
            agg_stats["rmse_ci_lower"] = rmse_low
            agg_stats["rmse_ci_upper"] = rmse_high
        # -----------------------------------------------

        # 2. Exact Match (ICO level)
        exact_matches = []
        if not scorable_df.empty:
            groups = scorable_df.groupby(self.id_cols)
            for _, group in groups:
                is_perfect = group['category'].isin(['TP', 'TN']).all()
                exact_matches.append(1 if is_perfect else 0)
        
        agg_stats['exact_match'] = np.mean(exact_matches) if exact_matches else 0.0

        # 3. Per-Field Metrics
        by_field = {}
        if not scorable_df.empty:
            for field_name, group in scorable_df.groupby('field'):
                by_field[field_name] = self._compute_stats(group)

        return {
            "aggregated": agg_stats,
            "by_field": by_field
        }

def calculate_metrics(extractions: List[Dict], gold_standard: List[Dict], subset: str = None) -> Dict[str, Any]:
    gold_filtered, pred_filtered = _filter_subset(gold_standard, extractions, subset)
    evaluator = Evaluator(gold_filtered, pred_filtered)
    metrics = evaluator.calculate_metrics()
    if subset:
        metrics["subset"] = subset
    return metrics