"""Concentration vs success rate analysis with statistics and visualisations.

Ported from analyse_concentration_success.py. Operates on in-memory DataFrames
instead of reading/writing intermediate CSVs.
"""

import os

import click
import numpy as np
import pandas as pd
from scipy import stats

from .plots import COLORS, MATCH_CATEGORIES, MISMATCH_REASONS, setup_plot_style
from .stats import create_concentration_bins

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def determine_reason(row):
    """Determine reason for mismatch or match status for a single row."""
    if pd.isna(row.get('matching')):
        return None

    matching = bool(row['matching'])
    match_type = row.get('match_type', None)

    if matching:
        if match_type == 'full':
            return 'Complete agreement between Sanger and Nanopore'
        elif match_type == 'partial':
            genus_count = row.get('genus_match_count', 0)
            if pd.notna(genus_count) and genus_count > 0:
                return f'Partial match with {int(genus_count)} genus-level matches'
            else:
                return 'Partial species match'
        else:
            return 'Match'

    # Mismatch - check QC first
    qc = row.get('qc', None)
    if pd.notna(qc):
        if isinstance(qc, bool) and qc is False:
            return 'QC Failed'
        elif isinstance(qc, str) and qc.lower() in ['false', 'fail', 'failed']:
            return 'QC Failed'

    # Then check for inhibition
    dilution_test = row.get('dilution_test', False)
    proteinase_k_test = row.get('proteinase_k_test', False)

    if isinstance(dilution_test, str):
        dilution_test = dilution_test.lower() == 'true'
    if isinstance(proteinase_k_test, str):
        proteinase_k_test = proteinase_k_test.lower() == 'true'

    if dilution_test is True or proteinase_k_test is True:
        return 'Inhibition'

    return 'Contamination'


def assign_match_category(df):
    """Add a ``match_category`` column that splits matches into 3 tiers.

    Categories for matches (matching==1):
      - Species Match: nanopore_missing_count == 0 (all sanger species found at species level)
      - Genus Match: sanger_undetected_count == 0 (all detected at genus+, some only genus)
      - Partial Match: some sanger species undetected even at genus level

    Non-matches keep their ``reason`` value (QC Failed / Inhibition / Contamination).
    """
    result = df.copy()
    conditions = [
        (result['matching'] == 1) & (result['nanopore_missing_count'].fillna(0) == 0),
        (result['matching'] == 1) & (result['sanger_undetected_count'].fillna(0) == 0),
        (result['matching'] == 1),
    ]
    choices = ['Species Match', 'Genus Match', 'Partial Match']
    result['match_category'] = np.select(conditions, choices, default='')
    # For non-matches, use the reason column
    mask_non_match = result['matching'] == 0
    result.loc[mask_non_match, 'match_category'] = result.loc[mask_non_match, 'reason']
    return result


def print_read_distributions(converged_df, mongo_data):
    """Print min/max/mean/median for total reads, top-hit abundance, and estimated counts.

    Excludes controls (positive & negative) and QC-failed samples.
    """

    click.echo("\n" + "-" * 80)
    click.echo("READ & TOP-HIT DISTRIBUTIONS (excl. controls & QC failures)")
    click.echo("-" * 80)

    # Filter out controls and QC-failed samples
    df = converged_df.copy()
    df = df[~df['sample_type'].str.contains('negative control', case=False, na=False)]
    df = df[~df['sample_type'].str.contains('positive control', case=False, na=False)]
    df = df[df['reason'] != 'QC Failed']
    valid_sids = set(df['sample_id'])

    # 1. Total reads per sample
    reads = df['number_of_reads'].dropna()
    if len(reads) > 0:
        click.echo(f"\nTotal reads per sample (n={len(reads)}):")
        click.echo(f"  Min:    {reads.min():.0f}")
        click.echo(f"  Max:    {reads.max():.0f}")
        click.echo(f"  Mean:   {reads.mean():.1f}")
        click.echo(f"  Median: {reads.median():.1f}")
    else:
        click.echo("\nNo read count data available")

    # 2. Top-hit abundance and estimated counts per sample
    top_abundances = []
    top_est_counts = []

    for sid, doc in mongo_data.items():
        if sid not in valid_sids:
            continue
        hits = doc.get('taxonomic_data', {}).get('hits', [])
        if not hits:
            continue
        top_hit = max(hits, key=lambda h: float(h.get('abundance', 0)))
        top_abundances.append(float(top_hit.get('abundance', 0)))
        est = top_hit.get('estimated_counts')
        if est is not None:
            top_est_counts.append(float(est))

    if top_abundances:
        arr = np.array(top_abundances)
        click.echo(f"\nTop-hit abundance per sample (n={len(arr)}):")
        click.echo(f"  Min:    {arr.min():.2f}%")
        click.echo(f"  Max:    {arr.max():.2f}%")
        click.echo(f"  Mean:   {arr.mean():.2f}%")
        click.echo(f"  Median: {np.median(arr):.2f}%")

    if top_est_counts:
        arr = np.array(top_est_counts)
        click.echo(f"\nTop-hit estimated counts per sample (n={len(arr)}):")
        click.echo(f"  Min:    {arr.min():.0f}")
        click.echo(f"  Max:    {arr.max():.0f}")
        click.echo(f"  Mean:   {arr.mean():.1f}")
        click.echo(f"  Median: {np.median(arr):.1f}")
    elif top_abundances:
        click.echo("\nNo estimated counts data available in top hits")


def build_converged_dataframe(input_df, mongo_data, matching_df, correct_concentration=False):
    """Merge user CSV + MongoDB metadata + matching data into one DataFrame.

    Parameters
    ----------
    input_df : DataFrame
        User-provided CSV with sample_id, dilution_test, proteinase_k_test.
    mongo_data : dict[str, dict]
        Pre-fetched MongoDB documents keyed by sample_id.
    matching_df : DataFrame
        Output of matching.generate_matching().

    Returns
    -------
    DataFrame
        Converged dataset with all fields needed for analysis.
    """
    click.echo("\nBuilding converged metadata DataFrame...")

    # Extract metadata fields from mongo docs into a flat DataFrame
    meta_rows = []
    for sample_id, doc in mongo_data.items():
        metadata = doc.get('metadata', {})
        row = {
            'sample_id': sample_id,
            'sample_type': metadata.get('sample_type'),
            'material': metadata.get('material'),
            'sanger_expected_species': metadata.get('sanger_expected_species'),
            'library_concentration': metadata.get('library_concentration'),
            'dilution': metadata.get('dilution'),
            'spike_concentration': metadata.get('spike_concentration'),
            'sequencing_run_id': doc.get('sequencing_run_id'),
        }

        # Extract number_of_reads from nested nanoplot path
        try:
            row['number_of_reads'] = (
                doc['nanoplot']['processed']['nanostats']['number_of_reads']
            )
        except (KeyError, TypeError):
            row['number_of_reads'] = 0

        try:
            row['unprocessed_reads'] = (
                doc['nanoplot']['unprocessed']['nanostats']['number_of_reads']
            )
        except (KeyError, TypeError):
            row['unprocessed_reads'] = 0

        meta_rows.append(row)

    mongo_df = pd.DataFrame(meta_rows)

    # Strip input_df to only required columns to prevent collisions with
    # MongoDB metadata columns (e.g. library_concentration, material).
    input_df = input_df[['sample_id', 'dilution_test', 'proteinase_k_test']].copy()

    # Merge: input CSV -> mongo metadata -> matching
    merged = input_df.merge(mongo_df, on='sample_id', how='left')
    merged = merged.merge(matching_df, on='sample_id', how='left', suffixes=('', '_match'))

    # Coerce types
    merged['matching'] = pd.to_numeric(merged['matching'], errors='coerce')
    merged['library_concentration'] = pd.to_numeric(
        merged['library_concentration'], errors='coerce'
    )
    merged['number_of_reads'] = pd.to_numeric(merged['number_of_reads'], errors='coerce')
    merged['unprocessed_reads'] = pd.to_numeric(merged['unprocessed_reads'], errors='coerce')
    merged['sanger_undetected_count'] = pd.to_numeric(
        merged.get('sanger_undetected_count', 0), errors='coerce'
    )

    # Preserve raw library concentration before potential correction
    merged['raw_library_concentration'] = merged['library_concentration'].copy()

    # Optionally correct library concentration by the ratio of processed to unprocessed reads
    if correct_concentration:
        read_ratio = merged['number_of_reads'] / merged['unprocessed_reads']
        merged['library_concentration'] = merged['library_concentration'] * read_ratio

    # Generate reason column
    merged['reason'] = merged.apply(determine_reason, axis=1)

    # Assign match category (Species Match / Genus Match / Partial Match / mismatch reasons)
    merged = assign_match_category(merged)

    click.echo(f"Converged dataset: {len(merged)} samples")
    click.echo(f"  Missing matching data: {merged['matching'].isna().sum()}")

    return merged


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_statistics(df, concentration_col='library_concentration'):
    """Calculate concentration bin stats, correlations, and chi-square test."""

    click.echo("\n" + "=" * 80)
    click.echo("STATISTICAL ANALYSIS: CONCENTRATION VS SUCCESS RATE")
    click.echo("=" * 80)

    df_valid = df[df['matching'].notna()].copy()

    total_samples = len(df_valid)
    total_matches = df_valid['matching'].sum()
    overall_success = (total_matches / total_samples * 100) if total_samples > 0 else 0

    click.echo(f"\nOverall Success Rate: {overall_success:.2f}% ({int(total_matches)}/{total_samples})")

    # Match-type breakdown
    if 'match_category' in df_valid.columns:
        matched = df_valid[df_valid['matching'] == 1]
        if len(matched) > 0:
            click.echo("\n" + "-" * 80)
            click.echo("MATCH TYPE BREAKDOWN")
            click.echo("-" * 80)
            click.echo(f"\nMatch categories (n={len(matched)}):")
            for cat_name, _ in MATCH_CATEGORIES:
                count = (matched['match_category'] == cat_name).sum()
                if count > 0:
                    pct = count / len(matched) * 100
                    click.echo(f"  {cat_name}: {count} ({pct:.1f}%)")

    # Failure reason breakdown
    if 'reason' in df_valid.columns:
        click.echo("\n" + "-" * 80)
        click.echo("FAILURE REASON ANALYSIS")
        click.echo("-" * 80)

        failed = df_valid[df_valid['matching'] == 0]
        if len(failed) > 0:
            reason_counts = failed['reason'].value_counts()
            click.echo(f"\nFailure reasons (n={len(failed)}):")
            for reason, count in reason_counts.items():
                pct = count / len(failed) * 100
                click.echo(f"  {reason}: {count} ({pct:.1f}%)")

    # Bin statistics
    valid_data, bin_edges = create_concentration_bins(df_valid, concentration_col=concentration_col)

    click.echo(f"\n{'Concentration Range':<25} {'N':<8} {'Matches':<10} {'Success %':<12} {'95% CI'}")
    click.echo("-" * 80)

    bin_stats = []
    for conc_range in valid_data['conc_range'].cat.categories:
        subset = valid_data[valid_data['conc_range'] == conc_range]
        n = len(subset)
        matches = subset['matching'].sum()

        if n > 0:
            success_rate = matches / n * 100

            if n >= 5:
                ci = stats.binom.interval(0.95, n, matches / n)
                ci_lower = ci[0] / n * 100
                ci_upper = ci[1] / n * 100
                ci_str = f"({ci_lower:.1f}-{ci_upper:.1f})"
            else:
                ci_str = "N/A"

            click.echo(f"{conc_range:<25} {n:<8} {int(matches):<10} {success_rate:<12.2f} {ci_str}")
            bin_stats.append({
                'range': conc_range, 'n': n,
                'matches': matches, 'success_rate': success_rate,
            })

    # Correlations
    click.echo("\n" + "-" * 80)
    click.echo("CORRELATION ANALYSIS")
    click.echo("-" * 80)

    valid_for_corr = df_valid[df_valid[concentration_col].notna()]

    if len(valid_for_corr) > 2:
        pearson_r, pearson_p = stats.pearsonr(
            valid_for_corr[concentration_col], valid_for_corr['matching']
        )
        click.echo(f"Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.4f}")

        spearman_r, spearman_p = stats.spearmanr(
            valid_for_corr[concentration_col], valid_for_corr['matching']
        )
        click.echo(f"Spearman correlation: rho = {spearman_r:.3f}, p = {spearman_p:.4f}")

        pb_r, pb_p = stats.pointbiserialr(
            valid_for_corr['matching'], valid_for_corr[concentration_col]
        )
        click.echo(f"Point-biserial correlation: r = {pb_r:.3f}, p = {pb_p:.4f}")

        if 'number_of_reads' in df_valid.columns:
            reads_valid = df_valid[df_valid['number_of_reads'].notna()]
            if len(reads_valid) > 2:
                reads_pb_r, reads_pb_p = stats.pointbiserialr(
                    reads_valid['matching'], reads_valid['number_of_reads']
                )
                click.echo(
                    f"Point-biserial correlation (reads vs matching): "
                    f"r = {reads_pb_r:.3f}, p = {reads_pb_p:.4f}"
                )

    # Chi-square
    if len(bin_stats) > 1:
        click.echo("\n" + "-" * 80)
        click.echo("CHI-SQUARE TEST (Independence of concentration bin and success)")
        click.echo("-" * 80)

        contingency = valid_data.groupby('conc_range')['matching'].agg(['sum', 'count'])
        contingency['non_match'] = contingency['count'] - contingency['sum']
        contingency_table = contingency[['sum', 'non_match']].values

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        click.echo(f"Chi-square statistic: {chi2:.3f}")
        click.echo(f"p-value: {p_value:.4f}")
        click.echo(f"Degrees of freedom: {dof}")

        if p_value < 0.05:
            click.echo("Result: Significant association between concentration range and success (p < 0.05)")
        else:
            click.echo("Result: No significant association between concentration range and success (p >= 0.05)")

    return valid_data, bin_stats


# ---------------------------------------------------------------------------
# Visualisations (6 plots)
# ---------------------------------------------------------------------------

def create_concentration_boxplot_combined(df, output_dir, concentration_col='library_concentration', file_suffix=''):
    """Plot 1: Concentration distribution by match status and mismatch reason."""
    click.echo("Creating plot 1: Concentration distribution by match status...")

    fig, ax = plt.subplots(figsize=(14, 7))

    df_with_conc = df[(df[concentration_col].notna()) & (df['matching'].notna())].copy()
    if len(df_with_conc) == 0:
        click.echo("  No data to plot")
        plt.close()
        return None

    conc_data = []
    labels = []
    colors_list = []

    for cat_name, cat_color in MATCH_CATEGORIES + MISMATCH_REASONS:
        cat_samples = df_with_conc[df_with_conc['match_category'] == cat_name]
        if len(cat_samples) > 0:
            conc_data.append(cat_samples[concentration_col].values)
            labels.append(f'{cat_name}\n(n={len(cat_samples)})')
            colors_list.append(cat_color)

    if not conc_data:
        click.echo("  No data to plot")
        plt.close()
        return None

    bp = ax.boxplot(
        conc_data, tick_labels=labels, patch_artist=True,
        showmeans=True, widths=0.6,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color='darkred'),
        meanprops=dict(marker='D', markerfacecolor='darkblue', markersize=6, markeredgecolor='black'),
    )
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(conc_data):
        x_jitter = np.random.default_rng(42).normal(i + 1, 0.04, size=len(data))
        ax.scatter(x_jitter, data, color='black', s=15, alpha=0.4, zorder=3)

    ax.set_ylabel('Library Concentration (ng/uL)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"01_concentration_by_status{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_sample_distribution_combined(df, output_dir, file_suffix=''):
    """Plot 2: Bar chart showing all sample categories."""
    click.echo("Creating plot 2: Combined sample distribution...")

    fig, ax = plt.subplots(figsize=(12, 7))

    categories = []
    counts = []
    colors_list = []

    neg_controls = df[df['matching'].isna()]
    if len(neg_controls) > 0:
        categories.append('Negative\nControl')
        counts.append(len(neg_controls))
        colors_list.append(COLORS['negative_control'])

    for cat_name, cat_color in MATCH_CATEGORIES + MISMATCH_REASONS:
        cat_samples = df[df['match_category'] == cat_name]
        if len(cat_samples) > 0:
            categories.append(cat_name)
            counts.append(len(cat_samples))
            colors_list.append(cat_color)

    if not counts:
        click.echo("  No data to plot")
        plt.close()
        return None

    bars = ax.bar(range(len(categories)), counts, alpha=0.7,
                  edgecolor='black', linewidth=1.5, color=colors_list)

    total_samples = len(df)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / total_samples * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"02_sample_distribution{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_species_agreement_plot(df, output_dir, file_suffix=''):
    """Plot 3: Species count scatter plot (successful matches only)."""
    click.echo("Creating plot 3: Species agreement analysis...")

    fig, ax = plt.subplots(figsize=(10, 8))

    valid_matches = df[(df['matching'] == 1) & (df['match_type'].notna())].copy()
    if len(valid_matches) == 0:
        click.echo("  No valid matches to plot")
        plt.close()
        return None

    valid_matches['sanger_total'] = (
        valid_matches['sanger_missing_count'].fillna(0) +
        valid_matches['nanopore_missing_count'].fillna(0) + 1
    )
    valid_matches['nanopore_total'] = valid_matches['sanger_total']

    for match_type, color, label in [
        ('full', COLORS['full_match'], 'Full Match'),
        ('partial', COLORS['partial_match'], 'Partial Match'),
    ]:
        subset = valid_matches[valid_matches['match_type'] == match_type]
        if len(subset) > 0:
            jitter = 0.1
            sanger_jitter = subset['sanger_total'] + np.random.uniform(-jitter, jitter, len(subset))
            nanopore_jitter = subset['sanger_total'] + np.random.uniform(-jitter, jitter, len(subset))
            ax.scatter(sanger_jitter, nanopore_jitter,
                       alpha=0.6, s=80, color=color,
                       edgecolors='black', linewidth=0.5, label=label)

    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='Perfect agreement')

    ax.set_xlabel('Sanger Species Count', fontsize=12)
    ax.set_ylabel('Nanopore Species Count', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"03_species_agreement{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_mismatch_reads_plot(df, output_dir, file_suffix=''):
    """Plot 4: Read count boxplot by mismatch reason with outlier labels."""
    click.echo("Creating plot 4: Read count distribution by mismatch reason...")

    if 'number_of_reads' not in df.columns:
        click.echo("  No read count data available")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    mismatches = df[(df['matching'] == 0) & (df['reason'].notna())].copy()
    if len(mismatches) == 0:
        click.echo("  No mismatches to plot")
        plt.close()
        return None

    mismatches['number_of_reads'] = mismatches['number_of_reads'].fillna(0)

    reads_data = []
    labels = []
    colors_list = []
    sample_ids_by_reason = []

    for reason, color in MISMATCH_REASONS:
        subset = mismatches[mismatches['reason'] == reason]
        if len(subset) > 0:
            reads_data.append(subset['number_of_reads'].values)
            labels.append(f"{reason}\n(n={len(subset)})")
            colors_list.append(color)
            sample_ids_by_reason.append(subset[['sample_id', 'number_of_reads']].values)

    if not reads_data:
        click.echo("  No data to plot")
        plt.close()
        return None

    bp = ax.boxplot(
        reads_data, tick_labels=labels, patch_artist=True,
        showmeans=True, widths=0.6,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color='darkred'),
        meanprops=dict(marker='D', markerfacecolor='darkblue', markersize=6, markeredgecolor='black'),
    )
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(reads_data):
        x_jitter = np.random.default_rng(42).normal(i + 1, 0.04, size=len(data))
        ax.scatter(x_jitter, data, color='black', s=15, alpha=0.4, zorder=3)

    # Label outliers
    for i, (data, sample_info) in enumerate(zip(reads_data, sample_ids_by_reason)):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        for sample_id, reads in sample_info:
            if reads < lower_bound or reads > upper_bound:
                ax.text(i + 1.05, reads, sample_id,
                        fontsize=8, ha='left', va='center', alpha=0.7)

    ax.set_ylabel('Number of Reads', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"04_mismatch_reads{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_reads_by_category_plot(df, output_dir, file_suffix=''):
    """Plot 7: Read count distribution by match status and mismatch reason."""
    click.echo("Creating plot 7: Read count distribution by category...")

    fig, ax = plt.subplots(figsize=(14, 7))

    df_with_reads = df[(df['number_of_reads'].notna()) & (df['matching'].notna())].copy()
    if len(df_with_reads) == 0:
        click.echo("  No data to plot")
        plt.close()
        return None

    reads_data = []
    labels = []
    colors_list = []

    for cat_name, cat_color in MATCH_CATEGORIES + MISMATCH_REASONS:
        cat_samples = df_with_reads[df_with_reads['match_category'] == cat_name]
        if len(cat_samples) > 0:
            reads_data.append(cat_samples['number_of_reads'].values)
            labels.append(f'{cat_name}\n(n={len(cat_samples)})')
            colors_list.append(cat_color)

    if not reads_data:
        click.echo("  No data to plot")
        plt.close()
        return None

    bp = ax.boxplot(
        reads_data, tick_labels=labels, patch_artist=True,
        showmeans=True, widths=0.6,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color='darkred'),
        meanprops=dict(marker='D', markerfacecolor='darkblue', markersize=6, markeredgecolor='black'),
    )
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(reads_data):
        x_jitter = np.random.default_rng(42).normal(i + 1, 0.04, size=len(data))
        ax.scatter(x_jitter, data, color='black', s=15, alpha=0.4, zorder=3)

    ax.set_ylabel('Number of Reads', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"07_reads_by_category{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_dilution_test_plot(df, output_dir, file_suffix=''):
    """Plot 5: Dilution test results (1:1 vs 1:10 dilution)."""
    click.echo("Creating plot 5: Dilution test analysis...")

    if 'dilution' not in df.columns:
        click.echo("  No dilution data available")
        return None

    dilution_samples = df[df['dilution'].notna()].copy()
    if len(dilution_samples) == 0:
        click.echo("  No dilution test samples found")
        return None

    dilution_samples['base_sample'] = dilution_samples['sample_id'].apply(
        lambda x: str(x).split("_")[0].split("-")[0].rstrip("sp") if pd.notna(x) else None
    )

    click.echo(f"  Found {len(dilution_samples)} samples with dilution data")
    click.echo(f"  Dilution values: {dilution_samples['dilution'].unique()}")

    base_samples = dilution_samples['base_sample'].dropna().unique()
    if len(base_samples) == 0:
        click.echo("  No base samples found")
        return None

    complete_pairs = []
    for base in base_samples:
        pair = dilution_samples[dilution_samples['base_sample'] == base]
        dilutions = pair['dilution'].unique()
        if '1:1' in dilutions and '1:10' in dilutions:
            complete_pairs.append(base)

    if len(complete_pairs) == 0:
        click.echo("  No complete dilution pairs found")
        return None

    click.echo(f"  Found {len(complete_pairs)} complete pairs")

    fig, ax = plt.subplots(figsize=(16, 8))

    x_pos = []
    labels = []
    colors_1_1 = []
    colors_1_10 = []
    qc_failed_1_1 = []
    qc_failed_1_10 = []
    sanger_species = []
    bar_width = 0.35

    for i, base_sample in enumerate(complete_pairs):
        pair = dilution_samples[dilution_samples['base_sample'] == base_sample]
        sample_1_1 = pair[pair['dilution'] == '1:1']
        sample_1_10 = pair[pair['dilution'] == '1:10']

        if len(sample_1_1) == 0 or len(sample_1_10) == 0:
            continue

        x_pos.append(i)
        labels.append(base_sample)

        sanger_sp = sample_1_1.iloc[0].get('sanger_expected_species', 'Unknown')
        sanger_species.append(str(sanger_sp) if pd.notna(sanger_sp) else 'Unknown')

        for sample_df, color_list, qc_list in [
            (sample_1_1, colors_1_1, qc_failed_1_1),
            (sample_1_10, colors_1_10, qc_failed_1_10),
        ]:
            row = sample_df.iloc[0]
            qc_fail = False
            if 'qc' in row and pd.notna(row['qc']):
                if row['qc'] in [False, 'False', 'false', 'fail', 'failed']:
                    qc_fail = True
            qc_list.append(qc_fail)

            if pd.isna(row.get('matching')):
                color_list.append(COLORS['no_data'])
            else:
                cat = row.get('match_category', '')
                cat_colors = {name: color for name, color in MATCH_CATEGORIES + MISMATCH_REASONS}
                color_list.append(cat_colors.get(cat, COLORS['no_data']))

    if not x_pos:
        click.echo("  No plottable pairs found after filtering")
        plt.close()
        return None

    x_array = np.array(x_pos)

    ax.bar(x_array - bar_width / 2, [1] * len(x_pos), bar_width,
           label='1:1 Dilution', alpha=0.7, edgecolor='black', linewidth=1.5,
           color=colors_1_1)
    ax.bar(x_array + bar_width / 2, [1] * len(x_pos), bar_width,
           label='1:10 Dilution', alpha=0.7, edgecolor='black', linewidth=1.5,
           color=colors_1_10)

    for i, (qc_1_1, qc_1_10) in enumerate(zip(qc_failed_1_1, qc_failed_1_10)):
        if qc_1_1:
            ax.text(x_array[i] - bar_width / 2, 1.05, 'X', ha='center', va='bottom',
                    fontsize=16, fontweight='bold', color='red')
        if qc_1_10:
            ax.text(x_array[i] + bar_width / 2, 1.05, 'X', ha='center', va='bottom',
                    fontsize=16, fontweight='bold', color='red')

    for i, species in enumerate(sanger_species):
        ax.text(x_array[i], 1.15, species, ha='center', va='bottom',
                fontsize=9, rotation=0, style='italic')

    ax.set_ylabel('Sample', fontsize=12)
    ax.set_xlabel('Sample Pair', fontsize=12)
    ax.set_xticks(x_array)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=name, alpha=0.7)
        for name, color in MATCH_CATEGORIES + MISMATCH_REASONS
    ] + [
        Patch(facecolor=COLORS['no_data'], edgecolor='black', label='No Data', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=2)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"05_dilution_test_results{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_dilution_sample_distribution(df, output_dir, file_suffix=''):
    """Plot 6: Stacked bar chart comparing 1:1 vs 1:10 dilution sample counts."""
    click.echo("Creating plot 6: 1:1 vs 1:10 dilution sample distribution...")

    if 'dilution' not in df.columns:
        click.echo("  No dilution data available")
        return None

    dilution_samples = df[df['dilution'].notna()].copy()
    if len(dilution_samples) == 0:
        click.echo("  No dilution test samples found")
        return None

    dilution_samples['base_sample'] = dilution_samples['sample_id'].apply(
        lambda x: str(x).split("_")[0].split("-")[0].rstrip("sp") if pd.notna(x) else None
    )

    base_samples = dilution_samples['base_sample'].dropna().unique()
    complete_pairs = []
    for base in base_samples:
        pair = dilution_samples[dilution_samples['base_sample'] == base]
        dilutions = pair['dilution'].unique()
        if '1:1' in dilutions and '1:10' in dilutions:
            complete_pairs.append(base)

    if len(complete_pairs) == 0:
        click.echo("  No complete dilution pairs found")
        return None

    dilution_samples = dilution_samples[dilution_samples['base_sample'].isin(complete_pairs)]
    click.echo(f"  Found {len(complete_pairs)} complete pairs ({len(dilution_samples)} samples)")

    fig, ax = plt.subplots(figsize=(10, 6))

    samples_1_1 = dilution_samples[dilution_samples['dilution'] == '1:1'].copy()
    samples_1_10 = dilution_samples[dilution_samples['dilution'] == '1:10'].copy()

    categories = [name for name, _ in MATCH_CATEGORIES + MISMATCH_REASONS]
    colors_map = {name: color for name, color in MATCH_CATEGORIES + MISMATCH_REASONS}

    samples_1_1['category'] = samples_1_1['match_category']
    samples_1_10['category'] = samples_1_10['match_category']

    samples_1_1 = samples_1_1[samples_1_1['category'].notna()]
    samples_1_10 = samples_1_10[samples_1_10['category'].notna()]

    counts_1_1 = samples_1_1['category'].value_counts()
    counts_1_10 = samples_1_10['category'].value_counts()

    values_1_1 = [counts_1_1.get(cat, 0) for cat in categories]
    values_1_10 = [counts_1_10.get(cat, 0) for cat in categories]

    x = np.array([0, 1])
    bar_width = 0.5

    bottom_1_1 = 0
    bottom_1_10 = 0

    for i, cat in enumerate(categories):
        if values_1_1[i] > 0 or values_1_10[i] > 0:
            ax.bar(x[0], values_1_1[i], bar_width, bottom=bottom_1_1,
                   label=cat, color=colors_map[cat], alpha=0.7,
)
            ax.bar(x[1], values_1_10[i], bar_width, bottom=bottom_1_10,
                   color=colors_map[cat], alpha=0.7)

            if values_1_1[i] > 0:
                ax.text(x[0], bottom_1_1 + values_1_1[i] / 2, str(values_1_1[i]),
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            if values_1_10[i] > 0:
                ax.text(x[1], bottom_1_10 + values_1_10[i] / 2, str(values_1_10[i]),
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

            bottom_1_1 += values_1_1[i]
            bottom_1_10 += values_1_10[i]

    if bottom_1_1 > 0:
        ax.text(x[0], bottom_1_1 + 0.3, f'Total: {int(bottom_1_1)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    if bottom_1_10 > 0:
        ax.text(x[1], bottom_1_10 + 0.3, f'Total: {int(bottom_1_10)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['1:1 Dilution', '1:10 Dilution'], fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"06_dilution_sample_distribution{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_concentration_vs_reads_plot(df, output_dir, file_suffix=''):
    """Plot 9: Library prep concentration vs number of reads."""
    click.echo("Creating plot 9: Library prep concentration vs number of reads...")

    fig, ax = plt.subplots(figsize=(10, 7))

    plot_df = df[df['matching'].notna()].copy()
    plot_df = plot_df[
        plot_df['raw_library_concentration'].notna() & plot_df['number_of_reads'].notna()
    ]
    if len(plot_df) == 0:
        click.echo("  No data to plot")
        plt.close()
        return None

    for cat_name, cat_color in MATCH_CATEGORIES + MISMATCH_REASONS:
        cat_samples = plot_df[plot_df['match_category'] == cat_name]
        if len(cat_samples) > 0:
            ax.scatter(cat_samples['raw_library_concentration'], cat_samples['number_of_reads'],
                       color=cat_color, label=f'{cat_name} (n={len(cat_samples)})',
                       alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Library Concentration (ng/uL)', fontsize=12)
    ax.set_ylabel('Number of Reads', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"09_concentration_vs_reads{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_reads_removed_vs_concentration_plot(df, output_dir, file_suffix=''):
    """Plot 8: Proportion of reads removed vs raw library concentration."""
    click.echo("Creating plot 8: Proportion of reads removed vs concentration...")

    fig, ax = plt.subplots(figsize=(10, 7))

    plot_df = df[df['matching'].notna()].copy()
    if 'raw_library_concentration' not in plot_df.columns:
        click.echo("  No raw_library_concentration column available")
        plt.close()
        return None

    # Compute proportion of reads removed
    plot_df['number_of_reads'] = plot_df['number_of_reads'].fillna(0)
    plot_df = plot_df[
        (plot_df['unprocessed_reads'].notna()) & (plot_df['unprocessed_reads'] > 0)
    ]
    plot_df['proportion_removed'] = 1 - (plot_df['number_of_reads'] / plot_df['unprocessed_reads'])

    plot_df = plot_df[plot_df['raw_library_concentration'].notna()]
    if len(plot_df) == 0:
        click.echo("  No data to plot")
        plt.close()
        return None

    for cat_name, cat_color in MATCH_CATEGORIES + MISMATCH_REASONS:
        cat_samples = plot_df[plot_df['match_category'] == cat_name]
        if len(cat_samples) > 0:
            ax.scatter(cat_samples['raw_library_concentration'], cat_samples['proportion_removed'],
                       color=cat_color, label=f'{cat_name} (n={len(cat_samples)})',
                       alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Library Concentration (ng/uL)', fontsize=12)
    ax.set_ylabel('Proportion of Reads Removed', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"08_reads_removed_vs_concentration{file_suffix}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_concentration_analysis(converged_df, output_dir, file_suffix=''):
    """Run the full concentration analysis: filter, stats, plots, save CSV.

    Parameters
    ----------
    converged_df : DataFrame
        Output of build_converged_dataframe().
    output_dir : str
        Directory for output files.
    file_suffix : str
        Suffix appended to output filenames (e.g. '_corrected').
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # Save converged data (before filtering)
    csv_path = os.path.join(output_dir, "converged_metadata.csv")
    converged_df.to_csv(csv_path, index=False)
    click.echo(f"\nMerged data saved to '{csv_path}'")

    # Filter controls
    click.echo("\nFiltering control samples...")
    from .stats import filter_controls
    filtered_df, neg_removed, pos_removed = filter_controls(converged_df)
    click.echo(f"  Removed {neg_removed} negative controls, {pos_removed} positive controls")
    click.echo(f"  Remaining: {len(filtered_df)} samples")

    # Print contamination samples
    contamination = filtered_df[
        (filtered_df['matching'] == 0) & (filtered_df['reason'] == 'Contamination')
    ]
    if len(contamination) > 0:
        click.echo(f"\nContamination samples (n={len(contamination)}):")
        for sid in contamination['sample_id']:
            click.echo(f"  - {sid}")
    else:
        click.echo("\nNo samples classified as contamination.")

    # Statistics
    valid_data, bin_stats = calculate_statistics(filtered_df)

    # Visualisations
    click.echo("\nGenerating visualisations...")
    click.echo("=" * 80)

    created = []
    for plot_fn in [
        lambda: create_concentration_boxplot_combined(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_sample_distribution_combined(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_species_agreement_plot(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_mismatch_reads_plot(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_dilution_test_plot(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_dilution_sample_distribution(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_reads_by_category_plot(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_reads_removed_vs_concentration_plot(filtered_df, output_dir, file_suffix=file_suffix),
        lambda: create_concentration_vs_reads_plot(filtered_df, output_dir, file_suffix=file_suffix),
    ]:
        path = plot_fn()
        if path:
            created.append(path)

    click.echo(f"\nCreated {len(created)} visualisation files:")
    for fp in created:
        click.echo(f"  - {os.path.basename(fp)}")

    return filtered_df
