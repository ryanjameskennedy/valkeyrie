"""Shared statistical utilities for 16S validation analysis."""

import pandas as pd
import numpy as np


def parse_species_list(species_str):
    """Parse species string into standardized list.

    Handles various formats: comma-separated, semicolon-separated, or single species.
    """
    if pd.isna(species_str) or species_str == '' or species_str is None:
        return []

    species_str = str(species_str)

    for separator in [';', ',', '|']:
        if separator in species_str:
            species_list = [s.strip() for s in species_str.split(separator)]
            return [s for s in species_list if s]

    return [species_str.strip()]


def create_concentration_bins(df, concentration_col='library_concentration', bin_width=5):
    """Create fixed-width concentration bins for analysis.

    Returns (binned_df, bin_edges). The returned DataFrame has a 'conc_range' column.
    """
    valid_data = df[df[concentration_col].notna()].copy()

    if len(valid_data) == 0:
        return valid_data, None

    max_val = valid_data[concentration_col].max()
    bin_edges = np.arange(0, max_val + bin_width, bin_width)

    labels = [
        f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}"
        for i in range(len(bin_edges) - 1)
    ]

    valid_data['conc_range'] = pd.cut(
        valid_data[concentration_col],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return valid_data, bin_edges


def calculate_success_rate_ci(successes, total, confidence=0.95):
    """Calculate success rate with binomial confidence interval.

    Returns dict with 'rate', 'ci_lower', 'ci_upper'.
    """
    from scipy import stats

    if total == 0:
        return {'rate': 0, 'ci_lower': 0, 'ci_upper': 0}

    rate = successes / total * 100

    if total >= 5:
        ci = stats.binom.interval(confidence, total, successes / total)
        ci_lower = ci[0] / total * 100
        ci_upper = ci[1] / total * 100
    else:
        ci_lower = None
        ci_upper = None

    return {'rate': rate, 'ci_lower': ci_lower, 'ci_upper': ci_upper}


def filter_controls(df):
    """Filter out negative and positive control samples.

    Expects a 'sample_type' column. Returns filtered DataFrame.
    """
    initial = len(df)
    df = df[~df['sample_type'].str.contains('negative control', case=False, na=False)]
    after_neg = len(df)
    df = df[~df['sample_type'].str.contains('positive control', case=False, na=False)]
    after_pos = len(df)

    removed_neg = initial - after_neg
    removed_pos = after_neg - after_pos

    return df, removed_neg, removed_pos
