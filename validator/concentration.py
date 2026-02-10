"""Concentration vs success rate analysis with statistics and visualisations.

Ported from analyse_concentration_success.py. Operates on in-memory DataFrames
instead of reading/writing intermediate CSVs.
"""

import os

import click
import numpy as np
import pandas as pd
from scipy import stats

from .plots import COLORS, MISMATCH_REASONS, setup_plot_style
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


def build_converged_dataframe(input_df, mongo_data, matching_df):
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
        }

        # Extract number_of_reads from nested nanoplot path
        try:
            row['number_of_reads'] = (
                doc['nanoplot']['processed']['nanostats']['number_of_reads']
            )
        except (KeyError, TypeError):
            row['number_of_reads'] = None

        meta_rows.append(row)

    mongo_df = pd.DataFrame(meta_rows)

    # Merge: input CSV -> mongo metadata -> matching
    merged = input_df.merge(mongo_df, on='sample_id', how='left')
    merged = merged.merge(matching_df, on='sample_id', how='left', suffixes=('', '_match'))

    # Coerce types
    merged['matching'] = pd.to_numeric(merged['matching'], errors='coerce')
    merged['library_concentration'] = pd.to_numeric(
        merged['library_concentration'], errors='coerce'
    )
    merged['number_of_reads'] = pd.to_numeric(merged['number_of_reads'], errors='coerce')

    # Generate reason column
    merged['reason'] = merged.apply(determine_reason, axis=1)

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

def create_concentration_boxplot_combined(df, output_dir, concentration_col='library_concentration'):
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

    full_matches = df_with_conc[(df_with_conc['matching'] == 1) & (df_with_conc['match_type'] == 'full')]
    if len(full_matches) > 0:
        conc_data.append(full_matches[concentration_col].values)
        labels.append(f'Full Match\n(n={len(full_matches)})')
        colors_list.append(COLORS['full_match'])

    partial_matches = df_with_conc[(df_with_conc['matching'] == 1) & (df_with_conc['match_type'] == 'partial')]
    if len(partial_matches) > 0:
        conc_data.append(partial_matches[concentration_col].values)
        labels.append(f'Partial Match\n(n={len(partial_matches)})')
        colors_list.append(COLORS['partial_match'])

    mismatches = df_with_conc[df_with_conc['matching'] == 0]
    for reason, color in MISMATCH_REASONS:
        reason_samples = mismatches[mismatches['reason'] == reason]
        if len(reason_samples) > 0:
            conc_data.append(reason_samples[concentration_col].values)
            labels.append(f'{reason}\n(n={len(reason_samples)})')
            colors_list.append(color)

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

    ax.set_ylabel('Library Concentration (ng/uL)', fontsize=12)
    ax.set_title('Concentration Distribution by Match Status', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "01_concentration_by_status.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_sample_distribution_combined(df, output_dir):
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

    full_matches = df[(df['matching'] == 1) & (df['match_type'] == 'full')]
    if len(full_matches) > 0:
        categories.append('Full\nMatch')
        counts.append(len(full_matches))
        colors_list.append(COLORS['full_match'])

    partial_matches = df[(df['matching'] == 1) & (df['match_type'] == 'partial')]
    if len(partial_matches) > 0:
        partial_with_genus = partial_matches[partial_matches['genus_match_count'] > 0]
        partial_without_genus = partial_matches[partial_matches['genus_match_count'] == 0]

        if len(partial_with_genus) > 0:
            categories.append('Partial Match\n(with genus)')
            counts.append(len(partial_with_genus))
            colors_list.append(COLORS['partial_match'])

        if len(partial_without_genus) > 0:
            categories.append('Partial Match\n(species only)')
            counts.append(len(partial_without_genus))
            colors_list.append(COLORS['partial_match_species'])

    mismatches = df[df['matching'] == 0]
    for reason, color in MISMATCH_REASONS:
        reason_samples = mismatches[mismatches['reason'] == reason]
        if len(reason_samples) > 0:
            categories.append(reason)
            counts.append(len(reason_samples))
            colors_list.append(color)

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
    ax.set_title(f'Sample Distribution by Category (n={total_samples} total)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "02_sample_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_species_agreement_plot(df, output_dir):
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
    ax.set_title('Species Count Agreement (Successful Matches Only)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "03_species_agreement.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_mismatch_reads_plot(df, output_dir):
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
    ax.set_title('Read Count Distribution by Mismatch Reason',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "04_mismatch_reads.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_dilution_test_plot(df, output_dir):
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
            elif row['matching'] == 1:
                if row.get('match_type') == 'full':
                    color_list.append(COLORS['full_match'])
                else:
                    color_list.append(COLORS['partial_match'])
            else:
                if row.get('reason') == 'QC Failed':
                    color_list.append(COLORS['qc_failed'])
                elif row.get('reason') == 'Inhibition':
                    color_list.append(COLORS['inhibition'])
                else:
                    color_list.append(COLORS['contamination'])

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
    ax.set_title('Dilution Test Results (1:1 vs 1:10)\nX = QC Failed, Italic = Sanger Expected Species',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_array)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=COLORS['full_match'], edgecolor='black', label='Full Match', alpha=0.7),
        Patch(facecolor=COLORS['partial_match'], edgecolor='black', label='Partial Match', alpha=0.7),
        Patch(facecolor=COLORS['qc_failed'], edgecolor='black', label='QC Failed', alpha=0.7),
        Patch(facecolor=COLORS['inhibition'], edgecolor='black', label='Inhibition', alpha=0.7),
        Patch(facecolor=COLORS['contamination'], edgecolor='black', label='Contamination', alpha=0.7),
        Patch(facecolor=COLORS['no_data'], edgecolor='black', label='No Data', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=2)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "05_dilution_test_results.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_dilution_sample_distribution(df, output_dir):
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

    categories = ['Full Match', 'Partial Match', 'QC Failed', 'Inhibition', 'Contamination']
    colors_map = {
        'Full Match': COLORS['full_match'],
        'Partial Match': COLORS['partial_match'],
        'QC Failed': COLORS['qc_failed'],
        'Inhibition': COLORS['inhibition'],
        'Contamination': COLORS['contamination'],
    }

    def categorize_sample(row):
        if pd.isna(row.get('matching')):
            return None
        elif row['matching'] == 1:
            return 'Full Match' if row.get('match_type') == 'full' else 'Partial Match'
        else:
            return row.get('reason', 'Unknown')

    samples_1_1['category'] = samples_1_1.apply(categorize_sample, axis=1)
    samples_1_10['category'] = samples_1_10.apply(categorize_sample, axis=1)

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
                   edgecolor='black', linewidth=1.5)
            ax.bar(x[1], values_1_10[i], bar_width, bottom=bottom_1_10,
                   color=colors_map[cat], alpha=0.7,
                   edgecolor='black', linewidth=1.5)

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
    ax.set_title('Sample Distribution: 1:1 vs 1:10 Dilution',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "06_dilution_sample_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_concentration_analysis(converged_df, output_dir):
    """Run the full concentration analysis: filter, stats, 6 plots, save CSV.

    Parameters
    ----------
    converged_df : DataFrame
        Output of build_converged_dataframe().
    output_dir : str
        Directory for output files.
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
        lambda: create_concentration_boxplot_combined(filtered_df, output_dir),
        lambda: create_sample_distribution_combined(filtered_df, output_dir),
        lambda: create_species_agreement_plot(filtered_df, output_dir),
        lambda: create_mismatch_reads_plot(filtered_df, output_dir),
        lambda: create_dilution_test_plot(filtered_df, output_dir),
        lambda: create_dilution_sample_distribution(filtered_df, output_dir),
    ]:
        path = plot_fn()
        if path:
            created.append(path)

    click.echo(f"\nCreated {len(created)} visualisation files:")
    for fp in created:
        click.echo(f"  - {os.path.basename(fp)}")

    return filtered_df
