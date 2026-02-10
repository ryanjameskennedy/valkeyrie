"""Material type vs success rate analysis with contamination focus.

Ported from analyse_material_success.py. Uses pre-fetched MongoDB data
instead of per-sample queries.
"""

import os

import click
import numpy as np
import pandas as pd
from scipy import stats

from .plots import COLORS, setup_plot_style
from .stats import calculate_success_rate_ci

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_material_statistics(df, material_column='material'):
    """Calculate material type vs success rate stats, chi-square, Cramer's V."""

    click.echo("\n" + "=" * 80)
    click.echo(f"STATISTICAL ANALYSIS: {material_column.upper()} VS SUCCESS RATE")
    click.echo("=" * 80)

    total_samples = len(df)
    total_matches = df['matching'].sum()
    overall_success = (total_matches / total_samples * 100) if total_samples > 0 else 0

    click.echo(f"\nOverall Success Rate: {overall_success:.2f}% ({int(total_matches)}/{total_samples})")

    material_counts = df[material_column].value_counts()
    click.echo(f"\nNumber of unique {material_column} types: {len(material_counts)}")
    click.echo(f"\n{material_column.capitalize()} distribution:")
    for material, count in material_counts.items():
        click.echo(f"  {material}: {count} samples")

    click.echo(
        f"\n{material_column.capitalize():<30} {'N':<8} {'Matches':<10} "
        f"{'Success %':<12} {'95% CI'}"
    )
    click.echo("-" * 80)

    material_stats = []
    for material in material_counts.index:
        subset = df[df[material_column] == material]
        n = len(subset)
        matches = subset['matching'].sum()

        if n > 0:
            ci_data = calculate_success_rate_ci(matches, n)
            ci_str = (
                f"({ci_data['ci_lower']:.1f}-{ci_data['ci_upper']:.1f})"
                if ci_data['ci_lower'] is not None else "N/A"
            )

            click.echo(f"{material:<30} {n:<8} {int(matches):<10} {ci_data['rate']:<12.2f} {ci_str}")

            material_stats.append({
                'material': material,
                'n': n,
                'matches': matches,
                'success_rate': ci_data['rate'],
                'mean_concentration': (
                    subset['library_concentration'].mean()
                    if 'library_concentration' in subset.columns else None
                ),
                'mean_reads': (
                    subset['number_of_reads'].mean()
                    if 'number_of_reads' in subset.columns else None
                ),
            })

    # Chi-square test
    click.echo("\n" + "-" * 80)
    click.echo("CHI-SQUARE TEST")
    click.echo("-" * 80)

    contingency_df = df.groupby([material_column, 'matching']).size().unstack(fill_value=0)
    contingency_table = contingency_df.values

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    click.echo(f"Chi-square: {chi2:.3f}, p-value: {p_value:.4f}, dof: {dof}")

    n_total = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
    click.echo(f"Cramer's V: {cramers_v:.3f}")

    return material_stats, material_counts


# ---------------------------------------------------------------------------
# Contamination analysis (uses pre-fetched data)
# ---------------------------------------------------------------------------

def filter_df_for_contamination_analysis(df, mongo_data, contamination_material='cerebrospinalvatska',
                                         material_column='material'):
    """Analyse contamination patterns using pre-fetched MongoDB data.

    Unlike the original script, this does NOT query MongoDB per-sample.
    Instead it reads from the mongo_data dict already in memory.
    """
    if not mongo_data:
        click.echo(f"No MongoDB data available, skipping {contamination_material.upper()} contamination analysis")
        return None

    click.echo("\n" + "=" * 80)
    click.echo(f"{contamination_material.upper()} FLUID CONTAMINATION ANALYSIS")
    click.echo("=" * 80)

    contaminated_samples = df[
        df[material_column].str.contains(contamination_material, case=False, na=False)
    ]

    if len(contaminated_samples) == 0:
        click.echo(f"No {contamination_material} samples found")
        return None

    click.echo(f"\nFound {len(contaminated_samples)} {contamination_material} samples")

    contamination_data = []

    for idx, row in contaminated_samples.iterrows():
        sample_id = row['sample_id']
        doc = mongo_data.get(sample_id)

        if not doc:
            continue

        flagged_contaminants = doc.get('flagged_contaminants', [])
        taxonomic_hits = doc.get('taxonomic_data', {}).get('hits', [])

        if not flagged_contaminants or not taxonomic_hits:
            continue

        contaminant_counts = {}
        total_contaminant_reads = 0

        for hit in taxonomic_hits:
            species = hit.get('species', '')
            read_count = hit.get('read_count', 0)

            if species in flagged_contaminants:
                contaminant_counts[species] = contaminant_counts.get(species, 0) + read_count
                total_contaminant_reads += read_count

        contamination_data.append({
            'sample_id': sample_id,
            'matching': row['matching'],
            'match_type': row.get('match_type'),
            'reason': row.get('reason'),
            'total_contaminant_reads': total_contaminant_reads,
            'num_contaminant_species': len(contaminant_counts),
            'contaminant_species': '; '.join(contaminant_counts.keys()),
            'contaminant_counts': contaminant_counts,
        })

    if not contamination_data:
        click.echo(f"No contamination data available for {contamination_material.upper()} samples")
        return None

    contamination_df = pd.DataFrame(contamination_data)

    click.echo(f"{contamination_material.upper()} samples with contamination data: {len(contamination_df)}")
    click.echo(
        f"Samples with detected contaminants: "
        f"{len(contamination_df[contamination_df['total_contaminant_reads'] > 0])}"
    )

    click.echo("\nContamination summary:")
    click.echo(f"  Mean contaminant reads per sample: {contamination_df['total_contaminant_reads'].mean():.1f}")
    click.echo(f"  Median contaminant reads per sample: {contamination_df['total_contaminant_reads'].median():.1f}")
    click.echo(f"  Max contaminant reads: {contamination_df['total_contaminant_reads'].max():.0f}")

    all_contaminants = {}
    for _, crow in contamination_df.iterrows():
        for species, count in crow['contaminant_counts'].items():
            all_contaminants[species] = all_contaminants.get(species, 0) + count

    if all_contaminants:
        click.echo("\nMost common contaminants (by total reads):")
        sorted_contaminants = sorted(all_contaminants.items(), key=lambda x: x[1], reverse=True)
        for species, count in sorted_contaminants[:10]:
            click.echo(f"  {species}: {count} reads")

    if len(contamination_df) > 2:
        contamination_df_valid = contamination_df[contamination_df['matching'].notna()]
        if len(contamination_df_valid) > 2:
            corr, p_val = stats.pointbiserialr(
                contamination_df_valid['matching'],
                contamination_df_valid['total_contaminant_reads'],
            )
            click.echo(f"\nCorrelation between contamination and matching status:")
            click.echo(f"  Point-biserial r = {corr:.3f}, p = {p_val:.4f}")

    click.echo("=" * 80)
    return contamination_df


# ---------------------------------------------------------------------------
# Visualisations (5 plots)
# ---------------------------------------------------------------------------

def create_material_concentration_boxplot(df, material_stats, material_column, output_dir):
    """Plot 1: Concentration distribution by material (colored by success rate)."""
    click.echo("Creating plot 1: Concentration distribution by material...")

    fig, ax = plt.subplots(figsize=(14, 7))

    if 'library_concentration' not in df.columns:
        click.echo("  No concentration data available")
        plt.close()
        return None

    material_stats_sorted = sorted(material_stats, key=lambda x: x['success_rate'], reverse=True)
    materials_ordered = [t['material'] for t in material_stats_sorted]

    conc_data = []
    conc_labels = []

    for material in materials_ordered:
        material_data = df[df[material_column] == material]['library_concentration'].dropna()
        if len(material_data) > 0:
            conc_data.append(material_data.values)
            conc_labels.append(material)

    if not conc_data:
        click.echo("  No data to plot")
        plt.close()
        return None

    bp = ax.boxplot(conc_data, tick_labels=conc_labels, patch_artist=True,
                    showmeans=True, widths=0.6)

    for i, patch in enumerate(bp['boxes']):
        material = conc_labels[i]
        material_success = next(t['success_rate'] for t in material_stats if t['material'] == material)
        color = plt.cm.RdYlGn(material_success / 100)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel(material_column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Library Concentration (ng/uL)', fontsize=12)
    ax.set_title(f'Concentration Distribution by {material_column.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(conc_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"01_{material_column}_concentration_boxplot.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_material_reads_boxplot(df, material_stats, material_column, output_dir):
    """Plot 2: Read count distribution by material."""
    click.echo("Creating plot 2: Read count distribution by material...")

    if 'number_of_reads' not in df.columns:
        click.echo("  No read count data available")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))

    material_stats_sorted = sorted(material_stats, key=lambda x: x['success_rate'], reverse=True)
    materials_ordered = [t['material'] for t in material_stats_sorted]

    reads_data = []
    reads_labels = []

    for material in materials_ordered:
        material_data = df[df[material_column] == material]['number_of_reads'].fillna(0)
        if len(material_data) > 0:
            reads_data.append(material_data.values)
            reads_labels.append(material)

    if not reads_data:
        click.echo("  No data to plot")
        plt.close()
        return None

    bp = ax.boxplot(reads_data, tick_labels=reads_labels, patch_artist=True,
                    showmeans=True, widths=0.6)

    for i, patch in enumerate(bp['boxes']):
        material = reads_labels[i]
        material_success = next(t['success_rate'] for t in material_stats if t['material'] == material)
        color = plt.cm.RdYlGn(material_success / 100)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel(material_column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Number of Reads', fontsize=12)
    ax.set_title(f'Read Count Distribution by {material_column.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(reads_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"02_{material_column}_reads_boxplot.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_material_success_rates(df, material_stats, material_column, output_dir):
    """Plot 3: Bar plot of success rates by material with overall mean line."""
    click.echo("Creating plot 3: Success rates by material...")

    fig, ax = plt.subplots(figsize=(14, 7))

    material_stats_sorted = sorted(material_stats, key=lambda x: x['success_rate'], reverse=True)

    materials = [t['material'] for t in material_stats_sorted]
    success_rates = [t['success_rate'] for t in material_stats_sorted]
    n_samples = [t['n'] for t in material_stats_sorted]

    colors = [plt.cm.RdYlGn(sr / 100) for sr in success_rates]

    bars = ax.bar(range(len(materials)), success_rates, alpha=0.7,
                  edgecolor='black', color=colors)

    for i, (bar, n) in enumerate(zip(bars, n_samples)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                f'n={n}', ha='center', va='bottom', fontsize=9)

    overall_mean = df['matching'].mean() * 100
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
               label=f'Overall: {overall_mean:.1f}%', alpha=0.7)

    ax.set_xlabel(material_column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Success Rate by {material_column.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(materials)))
    ax.set_xticklabels(materials, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"03_{material_column}_success_rates.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_contamination_heatmap(df, material_column, output_dir):
    """Plot 4: Crosstab heatmap of material vs mismatch reason."""
    click.echo("Creating plot 4: Contamination heatmap by material...")

    mismatches = df[df['matching'] == 0]
    if len(mismatches) == 0:
        click.echo("  No mismatches to plot")
        return None

    crosstab = pd.crosstab(mismatches[material_column], mismatches['reason'])
    if crosstab.empty:
        click.echo("  No data to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Samples'},
                linewidths=0.5, ax=ax)

    ax.set_xlabel('Mismatch Reason', fontsize=12)
    ax.set_ylabel(material_column.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Mismatch Reasons by {material_column.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"04_{material_column}_contamination_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_material_contamination_plot(contamination_df, output_dir,
                                       contamination_material='cerebrospinalvatska'):
    """Plot 5: 4-panel detailed contamination analysis."""
    click.echo("Creating plot 5: Contamination analysis...")

    if contamination_df is None or len(contamination_df) == 0:
        click.echo("  No contamination data available")
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Contaminant reads by match status
    match_groups = contamination_df.groupby('matching')['total_contaminant_reads'].apply(list)

    if len(match_groups) > 0:
        data_to_plot = []
        labels = []
        colors_list = []

        for match_val in [0, 1]:
            if match_val in match_groups.index:
                data_to_plot.append(match_groups[match_val])
                labels.append('Mismatch' if match_val == 0 else 'Match')
                colors_list.append(COLORS['mismatch'] if match_val == 0 else COLORS['match'])

        if data_to_plot:
            bp = ax1.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                             showmeans=True, widths=0.6)
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax1.set_ylabel('Total Contaminant Reads', fontsize=11)
            ax1.set_title('Contaminant Reads by Match Status', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Number of contaminant species
    if len(match_groups) > 0:
        species_data = []
        for match_val in [0, 1]:
            if match_val in match_groups.index:
                species_data.append(
                    contamination_df[contamination_df['matching'] == match_val]['num_contaminant_species'].values
                )

        if species_data:
            bp2 = ax2.boxplot(species_data, tick_labels=labels, patch_artist=True,
                              showmeans=True, widths=0.6)
            for patch, color in zip(bp2['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax2.set_ylabel('Number of Contaminant Species', fontsize=11)
            ax2.set_title('Contaminant Species Count by Match Status', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Top contaminants horizontal bar
    all_contaminants = {}
    for _, crow in contamination_df.iterrows():
        for species, count in crow['contaminant_counts'].items():
            all_contaminants[species] = all_contaminants.get(species, 0) + count

    if all_contaminants:
        sorted_contaminants = sorted(all_contaminants.items(), key=lambda x: x[1], reverse=True)[:10]
        species_names = [s[0] for s in sorted_contaminants]
        species_counts = [s[1] for s in sorted_contaminants]

        ax3.barh(range(len(species_names)), species_counts, alpha=0.7,
                 color=COLORS['contamination'], edgecolor='black')
        ax3.set_yticks(range(len(species_names)))
        ax3.set_yticklabels(species_names, fontsize=9)
        ax3.set_xlabel('Total Read Count', fontsize=11)
        ax3.set_title(f'Top 10 Contaminant Species ({contamination_material})',
                       fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

    # Panel 4: Scatter of contaminant reads vs matching
    samples_with_contam = contamination_df[contamination_df['total_contaminant_reads'] > 0]

    if len(samples_with_contam) > 0:
        for match_val, color, label in [
            (0, COLORS['mismatch'], 'Mismatch'),
            (1, COLORS['match'], 'Match'),
        ]:
            subset = samples_with_contam[samples_with_contam['matching'] == match_val]
            if len(subset) > 0:
                ax4.scatter(subset['num_contaminant_species'],
                            subset['total_contaminant_reads'],
                            alpha=0.6, s=80, color=color,
                            edgecolors='black', linewidth=0.5, label=label)

        ax4.set_xlabel('Number of Contaminant Species', fontsize=11)
        ax4.set_ylabel('Total Contaminant Reads', fontsize=11)
        ax4.set_title('Contamination Load vs Species Diversity', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"05_{contamination_material}_contamination_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_material_analysis(converged_df, mongo_data, output_dir,
                          material_column='material',
                          contamination_material='cerebrospinalvatska'):
    """Run the full material analysis: filter, stats, 5 plots, save CSV.

    Parameters
    ----------
    converged_df : DataFrame
        Filtered converged data (controls already removed).
    mongo_data : dict[str, dict]
        Pre-fetched MongoDB documents keyed by sample_id.
    output_dir : str
        Directory for output files.
    material_column : str
        Column name for material grouping.
    contamination_material : str
        Specific material type for contamination analysis.
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # Filter to samples with valid material and matching data
    df = converged_df[
        (converged_df[material_column].notna()) & (converged_df['matching'].notna())
    ].copy()
    click.echo(f"Material analysis dataset: {len(df)} samples with valid {material_column} and matching data")

    if len(df) == 0:
        click.echo("No samples to analyse for material analysis.")
        return

    # Statistics
    material_stats, material_counts = calculate_material_statistics(df, material_column=material_column)

    # Contamination analysis
    contamination_df = filter_df_for_contamination_analysis(
        df, mongo_data,
        contamination_material=contamination_material,
        material_column=material_column,
    )

    # Visualisations
    click.echo("\nGenerating visualisations...")
    created = []

    filepath = create_material_concentration_boxplot(df, material_stats, material_column, output_dir)
    if filepath:
        created.append(filepath)

    filepath = create_material_reads_boxplot(df, material_stats, material_column, output_dir)
    if filepath:
        created.append(filepath)

    filepath = create_material_success_rates(df, material_stats, material_column, output_dir)
    if filepath:
        created.append(filepath)

    filepath = create_contamination_heatmap(df, material_column, output_dir)
    if filepath:
        created.append(filepath)

    filepath = create_material_contamination_plot(contamination_df, output_dir, contamination_material)
    if filepath:
        created.append(filepath)

    click.echo(f"\nCreated {len(created)} visualisation files:")
    for fp in created:
        click.echo(f"  - {os.path.basename(fp)}")

    # Save contamination data
    if contamination_df is not None:
        contamination_output = os.path.join(
            output_dir, f'{contamination_material}_contamination_analysis.csv'
        )
        contamination_df.to_csv(contamination_output, index=False)
        click.echo(f"\n{contamination_material.upper()} contamination data saved to {contamination_output}")
