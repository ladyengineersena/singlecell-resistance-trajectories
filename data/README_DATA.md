# Data Directory

This directory contains synthetic and example data for the sc-resistance-trajectories pipeline.

## Structure

```
data/
├── synthetic/          # Generated synthetic single-cell datasets
└── README_DATA.md     # This file
```

## Synthetic Data

The `synthetic/` directory contains generated single-cell datasets that simulate:
- Multi-timepoint longitudinal samples
- Multiple patients with different resistance trajectories
- Various resistance mechanisms (EMT, PI3K/AKT, EGFR escape, etc.)
- Multi-modal data (scRNA-seq, scATAC-seq, CITE-seq)

### Generating Synthetic Data

```bash
python scripts/generate_synthetic_sc.py \
    --out data/synthetic \
    --n_patients 20 \
    --n_cells_per_patient 2000 \
    --n_timepoints 3
```

### Data Format

Synthetic data is stored as AnnData (`.h5ad`) files:
- One file per patient: `patient_{id}_timepoint_{tp}.h5ad`
- Or combined: `synthetic_combined.h5ad`

Each file contains:
- `X`: Count matrix (cells × genes)
- `obs`: Cell metadata (patient_id, timepoint, treatment, resistance_mechanism, etc.)
- `var`: Gene metadata
- `obsm`: Embeddings (PCA, UMAP)
- `layers`: Normalized counts, spliced/unspliced (for velocity)

## Real Patient Data

**⚠️ WARNING: Never store real patient data in this repository.**

If you have IRB-approved patient data:
1. Store it in a secure, encrypted location outside this repository
2. Ensure all PHI/PII is removed or anonymized
3. Use study IDs instead of patient identifiers
4. Follow your institution's data storage policies

## Data Sources (Public Datasets)

For research, consider these public data sources:

- **GEO (Gene Expression Omnibus)**: https://www.ncbi.nlm.nih.gov/geo/
- **SRA (Sequence Read Archive)**: https://www.ncbi.nlm.nih.gov/sra
- **Single Cell Portal**: https://singlecell.broadinstitute.org/
- **Human Cell Atlas**: https://www.humancellatlas.org/
- **CellXGene**: https://cellxgene.cziscience.com/

Search terms: "single cell cancer treatment scRNA-seq", "treatment resistance single cell"

## Data Requirements

### Minimum Required Fields

- `patient_id`: Unique identifier
- `timepoint`: Pre-treatment, on-treatment, progression
- `treatment_type`: Treatment modality
- `response_status`: Responder / Non-responder

### Optional Fields

- `treatment_dose`: Dose information
- `time_to_progression`: Days since treatment start
- `resistance_mechanism`: Ground truth labels
- `histology`: Tumor type
- `prior_therapies`: Previous treatments

## Data Quality

Before analysis, ensure:
- [ ] Quality control metrics calculated
- [ ] Doublets removed
- [ ] Batch effects identified
- [ ] Metadata complete and consistent
- [ ] No missing critical fields

