import numpy as np
import pandas as pd
from typing import Tuple


def _compute_stats(df_subset: pd.DataFrame) -> dict:
    """Compute precision/recall/F1 for a long-form evaluation DataFrame (F1 only use-case)."""
    df_subset = df_subset[df_subset['category'] != 'IGNORE']
    if df_subset.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    counts = df_subset['category'].value_counts()
    tp = counts.get('TP', 0)
    fp = counts.get('FP', 0)
    fn = counts.get('FN', 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compute_paired_difference_ci(
    zs_df: pd.DataFrame,
    fs_df: pd.DataFrame,
    metric: str = "f1",
    n_iterations: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap for the paired difference between Few-Shot and Zero-Shot.
    Resamples common PMCIDs in lockstep so documents remain paired across runs.
    Returns (point_estimate, lower, upper).
    Only F1 is supported; other metric values will be ignored.
    """
    if metric != "f1":
        metric = "f1"

    if zs_df is None or fs_df is None:
        return 0.0, 0.0, 0.0

    zs_df = zs_df[zs_df['category'] != 'IGNORE'].copy()
    fs_df = fs_df[fs_df['category'] != 'IGNORE'].copy()

    if zs_df.empty or fs_df.empty or 'pmcid' not in zs_df.columns or 'pmcid' not in fs_df.columns:
        return 0.0, 0.0, 0.0

    zs_df['pmcid'] = zs_df['pmcid'].astype(str)
    fs_df['pmcid'] = fs_df['pmcid'].astype(str)

    common_clusters = sorted(set(zs_df['pmcid']).intersection(fs_df['pmcid']))
    if not common_clusters:
        return 0.0, 0.0, 0.0

    baseline_zs = _compute_stats(zs_df).get("f1", 0.0)
    baseline_fs = _compute_stats(fs_df).get("f1", 0.0)
    point_estimate = baseline_fs - baseline_zs

    if len(common_clusters) < 2:
        return point_estimate, point_estimate, point_estimate

    grouped_zs = {pmcid: group for pmcid, group in zs_df.groupby('pmcid')}
    grouped_fs = {pmcid: group for pmcid, group in fs_df.groupby('pmcid')}

    deltas = []
    for _ in range(n_iterations):
        sampled = np.random.choice(common_clusters, size=len(common_clusters), replace=True)
        sample_zs = pd.concat([grouped_zs[c] for c in sampled])
        sample_fs = pd.concat([grouped_fs[c] for c in sampled])
        stat_zs = _compute_stats(sample_zs).get("f1", 0.0)
        stat_fs = _compute_stats(sample_fs).get("f1", 0.0)
        deltas.append(stat_fs - stat_zs)

    lower = float(np.percentile(deltas, (1 - ci) / 2 * 100))
    upper = float(np.percentile(deltas, (1 + ci) / 2 * 100))
    return float(point_estimate), lower, upper
