# eyrie-validator

CLI tool for 16S Nanopore sequencing validation analysis. Consolidates four standalone validation scripts into a single pipeline that compares Sanger and Nanopore results, analyses concentration and material effects on sequencing success, and generates statistical reports with visualisations.

## Installation

Install the heavy scientific dependencies via conda first (provides pre-built binaries and avoids C compilation issues), then pip-install the package on top:

```bash
conda create -n eyrie-validator python=3.11 numpy pandas scipy matplotlib seaborn
conda activate eyrie-validator
pip install .
```

## Usage

```bash
eyrie-validator validate -i samples.csv -o results/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | Input CSV file |
| `-o`, `--output` | *(required)* | Output directory |
| `--mongo-uri` | `mongodb://localhost:5811/` | MongoDB connection URI |
| `--mongo-db` | `eyrie` | MongoDB database name |
| `--mongo-collection` | `samples` | MongoDB collection name |
| `--material-column` | `material` | Column name for material grouping |
| `--contamination-material` | `cerebrospinalvätska` | Material type for contamination analysis |
| `--sequencing-run-id` | `None` | Filter spike analysis to a specific sequencing run ID |
| `-v`, `--verbose` | off | Enable verbose output |

### Input CSV format

The input CSV must contain these three columns:

```
sample_id,dilution_test,proteinase_k_test
SAMPLE001,False,False
SAMPLE002,True,False
```

All other metadata (material, concentration, species, read counts) is fetched from MongoDB.

## Outputs

All files are written to the directory specified by `-o`.

### CSVs

- `converged_metadata.csv` - merged dataset with input CSV + MongoDB metadata + Sanger/Nanopore matching results
- `{contamination_material}_contamination_analysis.csv` - per-sample contamination breakdown for the specified material type

### Concentration analysis plots (6)

1. `01_concentration_by_status.png` - concentration distribution by match status and mismatch reason
2. `02_sample_distribution.png` - bar chart of all sample categories
3. `03_species_agreement.png` - species count scatter (Sanger vs Nanopore)
4. `04_mismatch_reads.png` - read count distribution by mismatch reason
5. `05_dilution_test_results.png` - 1:1 vs 1:10 dilution paired results
6. `06_dilution_sample_distribution.png` - stacked bar comparing dilution outcomes

### Material analysis plots (10)

1. `01_{material}_concentration_boxplot.png` - concentration distribution by material type (with individual data point overlay for n=1 visibility)
2. `02_{material}_reads_boxplot.png` - read count distribution by material type (with individual data point overlay for n=1 visibility)
3. `03_{material}_success_rates.png` - success rate bar chart by material
4. `04_{material}_concentration_vs_reads_bubble.png` - mean concentration vs mean reads bubble plot (size = sample count, colour = success rate)
5. `05_{material}_contamination_heatmap.png` - material vs mismatch reason heatmap
6. `06_{contamination_material}_contamination_analysis.png` - stacked barplot of contaminant species abundance per sample
7. `07_{material}_failed_sample_investigation.png` - scatter of reads vs concentration for failed samples (colour = failure reason, marker shape = test type)
8. `08_{material}_multi_species_genus_detection.png` - genus-level detection proportion boxplot for multi-species samples (with individual data point overlay for n=1 visibility)
9. `09_spike_abundance_boxplot.png` - Agrobacterium fabrum spike abundance (%) boxplot by IC3/IC4 concentration and sample type (Validation vs Negative Control). Optionally filtered by `--sequencing-run-id`.
10. `10_negative_control_abundance.png` - stacked species abundance barplot for negative control samples (species ≤5% grouped as "Other")

## Integration with eyrie-popup

The `validator/` package is designed as a standalone module. To integrate it into the eyrie-popup application, copy it into `popup/validator/` and add the CLI entry point to the popup's command group.

## License

MIT
