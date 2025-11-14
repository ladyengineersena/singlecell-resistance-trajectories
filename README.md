# sc-resistance-trajectories

**Predicting Tumor Resistance Trajectories Using Multi-Modal Single-Cell Data**

This repository contains pipelines, notebooks, and synthetic data to develop models that forecast likely resistance mechanisms in tumors using single-cell multi-omic longitudinal data.

## ⚠️ ETHICS WARNING

**This is a research prototype only.**
- **No PHI (Protected Health Information) or real patient data** is stored in this repository
- Any use of patient data **must have IRB approval** and proper data transfer agreements (DTA)
- Model outputs are for **research and decision-support only** - they do not provide definitive treatment recommendations
- Always maintain human-in-the-loop for clinical decisions
- See [ETHICS.md](ETHICS.md) for detailed ethical guidelines

## Overview

This pipeline predicts which molecular/operational pathways a tumor is likely to use to develop treatment resistance (e.g., targeted therapy/AKT pathway, EGFR mutation escape, EMT-based escape, etc.) using multi-omic single-cell data.

### Key Features

- **Multi-modal support**: scRNA-seq, scATAC-seq, CITE-seq, spatial transcriptomics
- **Multiple prediction tasks**:
  - **Task A**: Classification of resistance mechanisms (multi-label)
  - **Task B**: Trajectory forecasting (pseudotime evolution)
  - **Task C**: Cell-level risk heatmaps
  - **Task D**: Time-to-resistance prediction (survival-like)
- **Explainability**: SHAP values, attention weights, pathway enrichment
- **Comprehensive validation**: Temporal validation, external cohorts, biological cross-checking

## Quickstart

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Demo Data

```bash
python scripts/generate_synthetic_sc.py --out data/synthetic --n_patients 20 --n_cells_per_patient 2000
```

### 3. Run Analysis Pipeline

Execute notebooks in order:

1. `notebooks/01_exploratory.ipynb` - Data exploration and QC
2. `notebooks/02_trajectory_inference.ipynb` - Trajectory and pseudotime analysis
3. `notebooks/03_feature_engineering.ipynb` - Feature extraction
4. `notebooks/04_modeling_and_eval.ipynb` - Model training and evaluation

Or run the full pipeline script:

```bash
python scripts/run_full_pipeline.py --data_dir data/synthetic --output_dir results/
```

## Project Structure

```
sc-resistance-trajectories/
├── data/
│   ├── synthetic/              # Synthetic demo single-cell datasets
│   └── README_DATA.md          # Data documentation
├── notebooks/
│   ├── 01_exploratory.ipynb
│   ├── 02_trajectory_inference.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling_and_eval.ipynb
├── src/
│   ├── preprocess/
│   │   ├── qc.py               # Quality control
│   │   └── batch_correction.py # Batch effect correction
│   ├── trajectory/
│   │   └── scvelo_wrapper.py   # RNA velocity and trajectory inference
│   ├── features/
│   │   └── feature_builder.py  # Feature engineering
│   ├── models/
│   │   ├── xgb_model.py        # XGBoost classifier
│   │   ├── mil_model.py        # Multiple Instance Learning
│   │   └── gnns/               # Graph Neural Networks
│   ├── evaluate/
│   │   └── metrics.py          # Evaluation metrics
│   └── utils.py                # Utility functions
├── scripts/
│   ├── generate_synthetic_sc.py
│   └── run_full_pipeline.py
├── requirements.txt
├── README.md
├── ETHICS.md
└── LICENSE
```

## Methodology

### A. Preprocessing
- Quality control (mitochondrial %, nFeature, doublet detection)
- Normalization (SCTransform or log1p)
- Batch correction (Harmony, Scanorama, ComBat)
- Feature selection (highly variable genes)

### B. Trajectory Inference
- Pseudotime analysis (Monocle3, Slingshot)
- RNA velocity (scVelo) for cell evolution direction
- Cell state annotation (Leiden clustering + marker-based)

### C. Feature Engineering
- Cell embeddings (PCA/UMAP, scVI/scANVI latent space)
- Cluster-level dynamics (frequency changes, expression slopes)
- Regulatory features (TF activity, chromatin accessibility)
- Cell-cell communication scores

### D. Predictive Modeling
- **Stage 1**: Trajectory modeling (scVelo + scVI)
- **Stage 2**: Sample-level prediction (XGBoost → Transformer/GNN)
- Multiple Instance Learning (MIL) for cell-to-sample aggregation
- Graph Neural Networks (GNN) for cell-cell relationships

### E. Evaluation
- Stratified k-fold cross-validation (patient-level)
- Temporal validation (train on earlier, test on later)
- External validation (independent cohorts)
- Metrics: AUC, PR-AUC, C-index, calibration curves

## Usage Examples

### Basic Workflow

```python
from src.preprocess.qc import run_qc
from src.trajectory.scvelo_wrapper import infer_trajectories
from src.features.feature_builder import FeatureBuilder
from src.models.xgb_model import XGBResistancePredictor

# 1. Preprocess
adata = run_qc(adata_raw)

# 2. Infer trajectories
adata = infer_trajectories(adata)

# 3. Extract features
feature_builder = FeatureBuilder(adata)
features = feature_builder.extract_sample_features()

# 4. Train model
model = XGBResistancePredictor()
model.train(features, labels)
predictions = model.predict(features_test)
```

## Data Requirements

### Input Format
- **AnnData** objects (`.h5ad` files) with:
  - `X`: count matrix (cells × genes)
  - `obs`: cell metadata (patient_id, timepoint, treatment, etc.)
  - `var`: gene metadata
  - `obsm`: embeddings (PCA, UMAP, etc.)
  - `layers`: normalized counts, spliced/unspliced (for velocity)

### Metadata Fields
- `patient_id`: Unique patient identifier
- `timepoint`: Pre-treatment, on-treatment, progression
- `treatment_type`: Treatment modality
- `treatment_dose`: Dose information
- `response_status`: Responder / Non-responder
- `time_to_progression`: Days since treatment start
- `resistance_mechanism`: Ground truth labels (if available)

## Contributing

This is a research prototype. Contributions should:
1. Maintain ethical standards (no PHI)
2. Include tests for new features
3. Update documentation
4. Follow code style guidelines

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{sc_resistance_trajectories,
  title = {sc-resistance-trajectories: Predicting Tumor Resistance Using Single-Cell Multi-Omic Data},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/ladyengineersena/singlecell-resistance-trajectories}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue.

---

**Disclaimer**: This software is provided for research purposes only. It is not intended for clinical use without proper validation and regulatory approval.

