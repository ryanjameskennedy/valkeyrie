"""Click CLI entry point for valkeyrie."""

import os
import sys

import click
import pandas as pd


@click.group()
@click.version_option(version=None, prog_name="valkeyrie",
                      package_name="valkeyrie")
def cli():
    """Valkeyrie - 16S Nanopore sequencing validation analysis.

    Compares Sanger and Nanopore results, analyses concentration and material
    effects on sequencing success, and generates statistical reports with
    visualisations.
    """
    pass


@cli.command()
@click.option('-i', '--input', 'input_csv', required=True,
              type=click.Path(exists=True),
              help='CSV file with columns: sample_id, dilution_test, proteinase_k_test')
@click.option('-o', '--output', 'output_dir', required=True,
              type=click.Path(),
              help='Output directory for results and plots')
@click.option('--mongo-uri', default='mongodb://localhost:5811/',
              help='MongoDB connection URI')
@click.option('--mongo-db', default='eyrie',
              help='MongoDB database name')
@click.option('--mongo-collection', default='samples',
              help='MongoDB collection name')
@click.option('--material-column', default='material',
              help='Column name for material grouping in material analysis')
@click.option('--contamination-material', default='cerebrospinalvätska',
              help='Specific material type for contamination analysis')
@click.option('--sequencing-run-id', default=None,
              help='Filter spike analysis to a specific sequencing run ID')
@click.option('--correct-concentration', is_flag=True,
              help='Correct library concentration by the ratio of processed to unprocessed reads')
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose output')
def validate(input_csv, output_dir, mongo_uri, mongo_db, mongo_collection,
             material_column, contamination_material, sequencing_run_id,
             correct_concentration, verbose):
    """Run full 16S validation pipeline.

    Takes a CSV with sample_id, dilution_test, proteinase_k_test columns,
    fetches all other data from MongoDB, and produces:

    \b
    - Sanger vs Nanopore matching analysis
    - Concentration vs success rate statistics + 6 plots
    - Material vs success rate statistics + 6 plots
    - converged_metadata.csv and contamination CSV
    """
    from .mongo import connect_to_mongo, fetch_samples_bulk
    from .matching import generate_matching
    from .concentration import build_converged_dataframe, print_read_distributions, run_concentration_analysis
    from .material import run_material_analysis

    click.echo("=" * 80)
    click.echo("VALKEYRIE - 16S NANOPORE VALIDATION ANALYSIS")
    click.echo("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Read input CSV
    click.echo(f"\nReading input CSV: {input_csv}")
    input_df = pd.read_csv(input_csv)

    required_cols = {'sample_id', 'dilution_test', 'proteinase_k_test'}
    missing = required_cols - set(input_df.columns)
    if missing:
        click.echo(f"ERROR: Missing required columns: {', '.join(missing)}")
        sys.exit(1)

    click.echo(f"  Loaded {len(input_df)} samples")

    if verbose:
        click.echo(f"  Columns: {list(input_df.columns)}")

    # 2. Connect to MongoDB and bulk fetch
    try:
        collection = connect_to_mongo(mongo_uri, mongo_db, mongo_collection)
    except Exception as e:
        click.echo(f"ERROR: Failed to connect to MongoDB: {e}")
        sys.exit(1)

    sample_ids = input_df['sample_id'].unique().tolist()
    mongo_data = fetch_samples_bulk(collection, sample_ids)

    # 3. Generate matching (Sanger vs Nanopore)
    matching_df = generate_matching(mongo_data)

    # 4. Build converged DataFrame
    converged_df = build_converged_dataframe(input_df, mongo_data, matching_df,
                                                correct_concentration=correct_concentration)

    # 4b. Print read and top-hit distributions
    print_read_distributions(converged_df, mongo_data)

    # 5. Run concentration analysis (saves CSV + 6 plots)
    click.echo("\n" + "=" * 80)
    click.echo("CONCENTRATION VS SUCCESS RATE ANALYSIS")
    click.echo("=" * 80)

    file_suffix = '_corrected' if correct_concentration else ''
    filtered_df = run_concentration_analysis(converged_df, output_dir,
                                             file_suffix=file_suffix)

    # 6. Run material analysis (saves 5 plots + contamination CSV)
    click.echo("\n" + "=" * 80)
    click.echo(f"{material_column.upper()} VS SUCCESS RATE ANALYSIS")
    click.echo("=" * 80)

    run_material_analysis(
        filtered_df, mongo_data, output_dir,
        material_column=material_column,
        contamination_material=contamination_material,
        full_df=converged_df,
        sequencing_run_id=sequencing_run_id,
    )

    click.echo("\n" + "=" * 80)
    click.echo("VALIDATION ANALYSIS COMPLETE")
    click.echo("=" * 80)
    click.echo(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    cli()
