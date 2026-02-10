"""Sanger vs Nanopore species comparison (matching) logic.

Ported from generate_matching_from_mongo.py to operate on pre-fetched data
instead of querying MongoDB directly.
"""

import click
import pandas as pd

from .stats import parse_species_list


def check_genus_match(sanger_species, nanopore_species):
    """Check if there's genus-level matching between species lists."""
    if not sanger_species or not nanopore_species:
        return False

    sanger_genera = {sp.split()[0].lower() for sp in sanger_species if sp.split()}
    nanopore_genera = {sp.split()[0].lower() for sp in nanopore_species if sp.split()}

    return len(sanger_genera & nanopore_genera) > 0


def count_genus_matches(sanger_species, nanopore_species):
    """Count how many genus-level matches exist between species lists."""
    if not sanger_species or not nanopore_species:
        return 0

    sanger_genera = {sp.split()[0].lower() for sp in sanger_species if sp.split()}
    nanopore_genera = {sp.split()[0].lower() for sp in nanopore_species if sp.split()}

    return len(sanger_genera & nanopore_genera)


def categorize_match(sanger_species, nanopore_species):
    """Categorize the match between Sanger and Nanopore results.

    Returns dict with: matching, match_type, genus_match_count,
    sanger_missing_count, nanopore_missing_count, reason.
    """
    sanger_set = {sp.lower() for sp in sanger_species}
    nanopore_set = {sp.lower() for sp in nanopore_species}

    # Handle empty cases
    if not sanger_set and not nanopore_set:
        return {
            'matching': False, 'match_type': 'mismatch', 'genus_match_count': 0,
            'sanger_missing_count': 0, 'nanopore_missing_count': 0,
            'reason': 'Both Sanger and Nanopore data missing',
        }
    if not sanger_set:
        return {
            'matching': False, 'match_type': 'mismatch', 'genus_match_count': 0,
            'sanger_missing_count': 0, 'nanopore_missing_count': len(nanopore_set),
            'reason': 'Sanger data missing',
        }
    if not nanopore_set:
        return {
            'matching': False, 'match_type': 'mismatch', 'genus_match_count': 0,
            'sanger_missing_count': len(sanger_set), 'nanopore_missing_count': 0,
            'reason': 'Nanopore data missing',
        }

    overlap = sanger_set & nanopore_set
    sanger_only = sanger_set - nanopore_set
    nanopore_only = nanopore_set - sanger_set

    sanger_missing_count = len(nanopore_only)
    nanopore_missing_count = len(sanger_only)

    if overlap:
        if not sanger_only and not nanopore_only:
            return {
                'matching': True, 'match_type': 'full', 'genus_match_count': 0,
                'sanger_missing_count': 0, 'nanopore_missing_count': 0,
                'reason': 'Complete agreement between Sanger and Nanopore',
            }
        else:
            genus_match_count = count_genus_matches(
                [s for s in sanger_species if s.lower() in sanger_only],
                [s for s in nanopore_species if s.lower() in nanopore_only],
            )
            genus_note = f"; {genus_match_count} genus-level matches" if genus_match_count > 0 else ""
            return {
                'matching': True, 'match_type': 'partial',
                'genus_match_count': genus_match_count,
                'sanger_missing_count': sanger_missing_count,
                'nanopore_missing_count': nanopore_missing_count,
                'reason': (
                    f'Partial match: {len(overlap)} species overlap; '
                    f'{nanopore_missing_count} Sanger species not found in Nanopore; '
                    f'{sanger_missing_count} Nanopore species not in Sanger{genus_note}'
                ),
            }

    # No species-level overlap - check genus level
    genus_match_count = count_genus_matches(sanger_species, nanopore_species)

    if genus_match_count > 0:
        return {
            'matching': True, 'match_type': 'partial',
            'genus_match_count': genus_match_count,
            'sanger_missing_count': sanger_missing_count,
            'nanopore_missing_count': nanopore_missing_count,
            'reason': f'Genus-level match only: {genus_match_count} genus matches (species differ)',
        }

    return {
        'matching': False, 'match_type': 'mismatch', 'genus_match_count': 0,
        'sanger_missing_count': sanger_missing_count,
        'nanopore_missing_count': nanopore_missing_count,
        'reason': (
            f'No overlap: Sanger expected {" & ".join(sanger_species)}; '
            f'Nanopore found {" & ".join(nanopore_species)}'
        ),
    }


def generate_matching(samples_data):
    """Generate matching DataFrame from pre-fetched MongoDB data.

    Parameters
    ----------
    samples_data : dict[str, dict]
        Mapping of sample_id to MongoDB document (from fetch_samples_bulk).

    Returns
    -------
    pandas.DataFrame
        Columns: sample_id, matching, match_type, genus_match_count,
        sanger_missing_count, nanopore_missing_count, qc, comments_qc
    """
    click.echo("=" * 80)
    click.echo("PROCESSING SAMPLES: SANGER vs NANOPORE MATCHING")
    click.echo("=" * 80 + "\n")

    def clean_string(s):
        if s is None:
            return None
        return str(s).replace(',', ';')

    results = []
    processed = 0
    matched = 0
    skipped_controls = 0

    for sample_id, doc in samples_data.items():
        metadata = doc.get('metadata', {})
        sample_type = metadata.get('sample_type', '')

        # QC fields
        qc = doc.get('qc', None)
        comments = doc.get('comments', {})
        comments_qc_list = comments.get('qc', []) if isinstance(comments, dict) else []

        if isinstance(comments_qc_list, list) and comments_qc_list:
            comments_qc = '; '.join([
                clean_string(item.get('comment', ''))
                for item in comments_qc_list
                if isinstance(item, dict) and item.get('comment')
            ])
        else:
            comments_qc = None

        # Skip negative controls
        if isinstance(sample_type, str) and 'negative control' in sample_type.lower():
            results.append({
                'sample_id': clean_string(sample_id),
                'matching': None, 'match_type': None, 'genus_match_count': None,
                'sanger_missing_count': None, 'nanopore_missing_count': None,
                'qc': clean_string(qc), 'comments_qc': comments_qc,
            })
            skipped_controls += 1
            continue

        sanger_expected = parse_species_list(metadata.get('sanger_expected_species', ''))

        nanopore_hits = doc.get('flagged_top_hits', [])
        if isinstance(nanopore_hits, list):
            nanopore_species = nanopore_hits
        else:
            nanopore_species = parse_species_list(nanopore_hits)

        match_result = categorize_match(sanger_expected, nanopore_species)

        results.append({
            'sample_id': clean_string(sample_id),
            'matching': match_result['matching'],
            'match_type': clean_string(match_result['match_type']),
            'genus_match_count': match_result['genus_match_count'],
            'sanger_missing_count': match_result['sanger_missing_count'],
            'nanopore_missing_count': match_result['nanopore_missing_count'],
            'qc': clean_string(qc),
            'comments_qc': comments_qc,
        })

        processed += 1
        if match_result['matching']:
            matched += 1

        if processed % 10 == 0:
            click.echo(f"Processed {processed} samples...")

    click.echo(f"\nProcessing complete:")
    click.echo(f"  Total samples: {processed + skipped_controls}")
    click.echo(f"  Negative controls (skipped): {skipped_controls}")
    click.echo(f"  Processed samples: {processed}")
    if processed > 0:
        click.echo(f"  Matching (True): {matched} ({matched/processed*100:.1f}% of processed)")
        click.echo(f"  Non-matching (False): {processed - matched} ({(processed-matched)/processed*100:.1f}% of processed)")

    results_df = pd.DataFrame(results)

    if processed > 0:
        click.echo(f"\nMatch type breakdown (excluding negative controls):")
        match_type_counts = results_df[results_df['match_type'].notna()]['match_type'].value_counts()
        for match_type, count in match_type_counts.items():
            click.echo(f"  {match_type}: {count} ({count/processed*100:.1f}%)")

        genus_matches = results_df[
            (results_df['genus_match_count'].notna()) & (results_df['genus_match_count'] > 0)
        ]
        if len(genus_matches) > 0:
            click.echo(f"\nGenus-level matches: {len(genus_matches)} samples")

    return results_df
