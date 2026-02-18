"""MongoDB connection and bulk data fetching for validation."""

import click
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def connect_to_mongo(uri, database, collection):
    """Establish connection to MongoDB and return collection handle.

    Raises on connection failure so callers can handle it.
    """
    click.echo(f"\nConnecting to MongoDB at {uri}...")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # Test connection
    client.admin.command('ping')
    click.echo("Successfully connected to MongoDB")

    db = client[database]
    coll = db[collection]

    doc_count = coll.count_documents({})
    click.echo(f"Database: {database}")
    click.echo(f"Collection: {collection}")
    click.echo(f"Total documents in collection: {doc_count}\n")

    return coll


def fetch_samples_bulk(collection, sample_ids):
    """Fetch all requested samples in a single $in query.

    Returns dict[sample_id, document] with the full document for each sample.
    Only the fields needed by downstream analysis are projected.
    """
    projection = {
        "sample_id": 1,
        "sample_name": 1,
        "sequencing_run_id": 1,
        "metadata": 1,
        "flagged_top_hits": 1,
        "qc": 1,
        "comments.qc": 1,
        "nanoplot.processed.nanostats.number_of_reads": 1,
        "nanoplot.unprocessed.nanostats.number_of_reads": 1,
        "flagged_contaminants": 1,
        "taxonomic_data.hits": 1,
        "_id": 0,
    }

    click.echo(f"Fetching {len(sample_ids)} samples from MongoDB (bulk query)...")

    cursor = collection.find(
        {"sample_id": {"$in": list(sample_ids)}},
        projection,
    )

    results = {}
    for doc in cursor:
        sid = doc.get("sample_id")
        if sid:
            results[sid] = doc

    click.echo(f"Retrieved {len(results)}/{len(sample_ids)} samples from MongoDB")
    missing = set(sample_ids) - set(results.keys())
    if missing:
        click.echo(f"  Missing samples: {len(missing)}")

    return results
