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

        contaminant_abundance = {}
        total_contaminant_abundance = 0

        for hit in taxonomic_hits:
            species = hit.get('species', '')
            abundance = hit.get('abundance', 0)

            if species in flagged_contaminants:
                contaminant_abundance[species] = contaminant_abundance.get(species, 0) + abundance
                total_contaminant_abundance += abundance

        contamination_data.append({
            'sample_id': sample_id,
            'sample_name': doc.get('sample_name', sample_id),
            'sequencing_run_id': doc.get('sequencing_run_id', ''),
            'matching': row['matching'],
            'match_type': row.get('match_type'),
            'reason': row.get('reason'),
            'total_contaminant_abundance': total_contaminant_abundance,
            'num_contaminant_species': len(contaminant_abundance),
            'contaminant_species': '; '.join(contaminant_abundance.keys()),
            'contaminant_abundance': contaminant_abundance,
        })

    if not contamination_data:
        click.echo(f"No contamination data available for {contamination_material.upper()} samples")
        return None

    contamination_df = pd.DataFrame(contamination_data)

    click.echo(f"{contamination_material.upper()} samples with contamination data: {len(contamination_df)}")
    click.echo(
        f"Samples with detected contaminants: "
        f"{len(contamination_df[contamination_df['total_contaminant_abundance'] > 0])}"
    )

    click.echo("\nContamination summary:")
    click.echo(f"  Mean contaminant reads per sample: {contamination_df['total_contaminant_abundance'].mean():.1f}")
    click.echo(f"  Median contaminant reads per sample: {contamination_df['total_contaminant_abundance'].median():.1f}")
    click.echo(f"  Max contaminant reads: {contamination_df['total_contaminant_abundance'].max():.0f}")

    all_contaminants = {}
    for _, crow in contamination_df.iterrows():
        for species, count in crow['contaminant_abundance'].items():
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
                contamination_df_valid['total_contaminant_abundance'],
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

    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    norm = Normalize(vmin=0, vmax=100)
    sm = cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=10)

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

    bars = ax.bar(range(len(materials)), success_rates, alpha=0.7,
                  edgecolor='black', color='steelblue')

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


def create_material_bubble_plot(df, material_stats, material_column, output_dir):
    """Plot 4: Bubble plot of mean concentration vs mean reads by material."""
    click.echo("Creating plot 4: Mean concentration vs mean read count bubble plot...")

    if 'number_of_reads' not in df.columns:
        click.echo("  No read count data available")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    material_stats_sorted = sorted(material_stats, key=lambda x: x['success_rate'], reverse=True)

    mean_concs = []
    mean_reads = []
    success_rates = []
    labels = []
    ns = []

    for info in material_stats_sorted:
        material = info['material']
        material_df = df[df[material_column] == material]
        mean_conc = material_df['library_concentration'].mean()
        mean_read = material_df['number_of_reads'].mean()

        if not pd.isna(mean_conc) and not pd.isna(mean_read):
            mean_concs.append(mean_conc)
            mean_reads.append(mean_read)
            success_rates.append(info['success_rate'])
            labels.append(material)
            ns.append(info['n'])

    if not mean_concs:
        click.echo("  No data to plot")
        plt.close()
        return None

    sizes = 100 + np.array(ns) ** 2 * 30

    scatter = ax.scatter(mean_concs, mean_reads,
                         s=sizes, alpha=0.6, linewidth=0,
                         c=success_rates, cmap='RdYlGn', vmin=0, vmax=100)

    from adjustText import adjust_text

    texts = []
    for i, label in enumerate(labels):
        texts.append(ax.text(mean_concs[i], mean_reads[i], label,
                             fontsize=9, alpha=0.7))
    adjust_text(texts, ax=ax)

    ax.set_xlabel('Mean Library Concentration (ng/\u00b5L)', fontsize=12)
    ax.set_ylabel('Mean Number of Reads', fontsize=12)
    ax.set_title(
        f'Mean Read Count vs Mean Concentration by {material_column.replace("_", " ").title()}\n'
        f'(bubble size = sample count, colour = success rate)',
        fontsize=14, fontweight='bold',
    )
    ax.grid(alpha=0.3)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(min(y_min, -y_max * 0.05), y_max)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"04_{material_column}_concentration_vs_reads_bubble.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_contamination_heatmap(df, material_column, output_dir):
    """Plot 5: Crosstab heatmap of material vs mismatch reason."""
    click.echo("Creating plot 5: Contamination heatmap by material...")

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
    filepath = os.path.join(output_dir, f"05_{material_column}_contamination_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


def create_material_contamination_plot(contamination_df, output_dir,
                                       contamination_material='cerebrospinalvatska'):
    """Plot 6: Stacked barplot of contaminant species abundance per sample."""
    click.echo("Creating plot 6: Contamination analysis...")

    if contamination_df is None or len(contamination_df) == 0:
        click.echo("  No contamination data available")
        return None

    # Sort samples by sequencing_run_id so runs are grouped together
    contamination_df = contamination_df.sort_values('sequencing_run_id')

    # Collect all unique contaminant species and build a read-count matrix
    all_species = set()
    for abundance in contamination_df['contaminant_abundance']:
        all_species.update(abundance.keys())

    # Sort species by total abundance (descending) so the most common are at
    # the bottom of the stack and appear first in the legend.
    sorted_species = []
    if all_species:
        species_totals = {}
        for species in all_species:
            species_totals[species] = sum(
                abundance.get(species, 0) for abundance in contamination_df['contaminant_abundance']
            )
        sorted_species = sorted(species_totals, key=species_totals.get, reverse=True)

    sample_names = contamination_df['sample_name'].values
    run_ids = contamination_df['sequencing_run_id'].values

    if sorted_species:
        matrix = np.array([
            [row['contaminant_abundance'].get(sp, 0) for sp in sorted_species]
            for _, row in contamination_df.iterrows()
        ])  # shape: (n_samples, n_species)
    else:
        matrix = np.zeros((len(sample_names), 0))

    # Pick a qualitative colormap with enough colours for species
    species_cmap = plt.get_cmap('tab20')
    n_species = len(sorted_species)
    species_colors = [species_cmap(i % 20) for i in range(n_species)]

    fig_width = max(10, len(sample_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x = np.arange(len(sample_names))
    bottoms = np.zeros(len(sample_names))

    for idx, species in enumerate(sorted_species):
        values = matrix[:, idx]
        ax.bar(x, values, bottom=bottoms, color=species_colors[idx], edgecolor='white',
               linewidth=0.3, label=species)
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=90, fontsize=8)
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Abundance', fontsize=12)
    ax.set_title(
        f'Contaminant Species Abundance per Sample ({contamination_material})',
        fontsize=14, fontweight='bold',
    )
    ax.grid(axis='y', alpha=0.3)

    # Colour x-axis tick labels by sequencing_run_id
    unique_runs = list(dict.fromkeys(run_ids))  # preserves order
    print("unique_runs", unique_runs)
    run_cmap = plt.get_cmap('tab10')
    run_color_map = {run: run_cmap(i % 10) for i, run in enumerate(unique_runs)}

    for label, run_id in zip(ax.get_xticklabels(), run_ids):
        label.set_color(run_color_map[run_id])

    # Place species legend outside the plot area
    if sorted_species:
        species_legend = ax.legend(
            fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left',
            borderaxespad=0, title='Species',
        )
        ax.add_artist(species_legend)

    # Add a second legend for sequencing run colours
    from matplotlib.patches import Patch
    run_patches = [Patch(facecolor=run_color_map[run], label=run) for run in unique_runs]
    ax.legend(
        handles=run_patches, fontsize=8, bbox_to_anchor=(1.02, 0), loc='lower left',
        borderaxespad=0, title='Sequencing Run',
    )

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"06_{contamination_material}_contamination_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    return filepath


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_material_analysis(converged_df, mongo_data, output_dir,
                          material_column='material',
                          contamination_material='cerebrospinalvätska'):
    """Run the full material analysis: filter, stats, 6 plots, save CSV.

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

    filepath = create_material_bubble_plot(df, material_stats, material_column, output_dir)
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
