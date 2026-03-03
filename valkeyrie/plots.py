"""Shared color constants and matplotlib setup for validation plots."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


# Consistent color scheme across all plots
COLORS = {
    'full_match':            '#44aa44',  # green
    'partial_match':         '#4169e1',  # blue (Genus Match)
    'partial_match_weak':    '#f0c430',  # yellow
    'partial_match_species': '#aadd88',
    'qc_failed':             '#D73027',  # red
    'inhibition':            '#ff6b6b',
    'contamination':         '#4169e1',  # blue (kept for back-compat)
    'mismatch':              '#E07B39',  # dark orange
    'negative_control':      '#808080',
    'match':                 '#44aa44',
    'no_data':               '#808080',
}

MATCH_CATEGORIES = [
    ('Species Match', COLORS['full_match']),         # green
    ('Genus Match',   COLORS['partial_match']),      # blue
    ('Partial Match', COLORS['partial_match_weak']), # yellow
]

MISMATCH_REASONS = [
    ('QC Failed',  COLORS['qc_failed']),   # red
    ('Inhibition', COLORS['inhibition']),  # coral
    ('Mismatch',   COLORS['mismatch']),    # dark orange
]


def setup_plot_style():
    """Configure matplotlib/seaborn defaults for all validation plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
