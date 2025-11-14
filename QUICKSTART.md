# Quick Start Guide

## Installation

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: Some optional packages may require additional installation:
   - `harmonypy` for Harmony batch correction
   - `scanorama` for Scanorama batch correction
   - `scvelo` for RNA velocity (recommended)
   - `torch` and `torch-geometric` for deep learning models

## Generate Synthetic Data

```bash
python scripts/generate_synthetic_sc.py --out data/synthetic --n_patients 20 --n_cells_per_patient 2000
```

This will create:
- Individual patient files: `data/synthetic/Patient_XXX_timepoint_YYY.h5ad`
- Combined file: `data/synthetic/synthetic_combined.h5ad`
- Patient metadata: `data/synthetic/patient_metadata.csv`

## Run Full Pipeline

```bash
python scripts/run_full_pipeline.py --data_dir data/synthetic --output_dir results
```

This will:
1. Load and preprocess data
2. Perform quality control
3. Normalize and correct batch effects
4. Infer trajectories
5. Extract features
6. Train prediction model
7. Save results

## Run Step-by-Step (Jupyter Notebooks)

1. **Exploratory Analysis:**
   ```bash
   jupyter notebook notebooks/01_exploratory.ipynb
   ```

2. **Trajectory Inference:**
   ```bash
   jupyter notebook notebooks/02_trajectory_inference.ipynb
   ```

3. **Feature Engineering:**
   ```bash
   jupyter notebook notebooks/03_feature_engineering.ipynb
   ```

4. **Modeling and Evaluation:**
   ```bash
   jupyter notebook notebooks/04_modeling_and_eval.ipynb
   ```

## Expected Outputs

After running the pipeline, you should have:

- `results/processed_data.h5ad` - Preprocessed single-cell data
- `results/sample_features.csv` - Extracted features
- `results/predictions.csv` - Model predictions
- `results/feature_importance.csv` - Feature importance scores
- `results/xgb_model.pkl` - Trained model

## Troubleshooting

### Import Errors
If you get import errors, make sure:
- Virtual environment is activated
- All packages in `requirements.txt` are installed
- You're running from the project root directory

### Data Not Found
If data files are not found:
- Run the synthetic data generator first
- Check that `data/synthetic/` directory exists
- Verify file paths in scripts/notebooks

### Memory Issues
For large datasets:
- Reduce `--n_cells_per_patient` when generating data
- Use batch processing
- Consider using a machine with more RAM

### Missing Optional Dependencies
Some features require optional packages:
- Harmony: `pip install harmonypy`
- Scanorama: `pip install scanorama`
- scVelo: `pip install scvelo`
- PyTorch: `pip install torch torch-geometric`

The pipeline will use fallback methods if optional packages are not available.

## Next Steps

1. Review the [README.md](README.md) for detailed documentation
2. Read [ETHICS.md](ETHICS.md) before using with patient data
3. Customize the pipeline for your specific use case
4. Add your own data (with proper IRB approval)

