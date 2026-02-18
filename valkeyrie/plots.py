"""Shared color constants and matplotlib setup for validation plots."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


# Consistent color scheme across all plots
COLORS = {
    'full_match': '#44aa44',
    'partial_match': '#88cc88',
    'partial_match_weak': '#f0c430',
    'partial_match_species': '#aadd88',
    'qc_failed': '#ffa500',
    'inhibition': '#ff6b6b',
    'contamination': '#4169e1',
    'negative_control': '#808080',
    'mismatch': '#D73027',
    'match': '#44aa44',
    'no_data': '#808080',
}

MATCH_CATEGORIES = [
    ('Species Match', COLORS['full_match']),
    ('Genus Match', COLORS['partial_match']),
    ('Partial Match', COLORS['partial_match_weak']),
]

MISMATCH_REASONS = [
    ('QC Failed', COLORS['qc_failed']),
    ('Inhibition', COLORS['inhibition']),
    ('Contamination', COLORS['contamination']),
]


def setup_plot_style():
    """Configure matplotlib/seaborn defaults for all validation plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
