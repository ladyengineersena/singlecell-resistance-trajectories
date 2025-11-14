"""
Full Pipeline Script

Runs the complete analysis pipeline from raw data to predictions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import run_qc, correct_batch
from src.trajectory import infer_trajectories
from src.features import FeatureBuilder
from src.models import XGBResistancePredictor
from src.evaluate import generate_evaluation_report
from src.utils import load_adata, save_adata, normalize_data, find_hvg

sc.settings.verbosity = 2


def main():
    parser = argparse.ArgumentParser(
        description='Run full resistance prediction pipeline'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing input data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help='Specific data file to use (if None, uses combined file)'
    )
    parser.add_argument(
        '--batch_correction',
        type=str,
        default='harmony',
        choices=['harmony', 'scanorama', 'combat', 'bbknn', 'none'],
        help='Batch correction method'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgb',
        choices=['xgb', 'mil'],
        help='Model type'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Single-Cell Resistance Trajectory Prediction Pipeline")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    data_dir = Path(args.data_dir)
    
    if args.data_file:
        data_file = data_dir / args.data_file
    else:
        data_file = data_dir / "synthetic_combined.h5ad"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    adata = load_adata(str(data_file))
    print(f"Loaded data: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    
    # 2. Quality control
    print("\n[2/7] Quality control...")
    adata, qc_stats = run_qc(adata, remove_doublets=True)
    print(f"After QC: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    
    # 3. Normalization
    print("\n[3/7] Normalization...")
    adata = normalize_data(adata, method='log1p')
    adata = find_hvg(adata, n_top_genes=2000)
    
    # 4. Batch correction
    if args.batch_correction != 'none':
        print(f"\n[4/7] Batch correction ({args.batch_correction})...")
        adata = correct_batch(adata, method=args.batch_correction, batch_key='patient_id')
    else:
        print("\n[4/7] Skipping batch correction...")
        sc.tl.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_pcs=50)
        sc.tl.umap(adata)
    
    # 5. Trajectory inference
    print("\n[5/7] Trajectory inference...")
    adata = infer_trajectories(
        adata,
        compute_velocity=True,
        compute_pseudotime=True
    )
    
    # Save processed data
    processed_file = output_dir / "processed_data.h5ad"
    save_adata(adata, str(processed_file))
    print(f"Saved processed data to {processed_file}")
    
    # 6. Feature engineering
    print("\n[6/7] Feature engineering...")
    feature_builder = FeatureBuilder(adata)
    
    # Extract sample-level features
    sample_features = feature_builder.extract_sample_features(
        include_clusters=True,
        include_trajectory=True
    )
    
    # Prepare labels
    if 'resistance_mechanism' in adata.obs.columns:
        # Get patient-level labels
        patient_labels = adata.obs.groupby('patient_id')['resistance_mechanism'].first().reset_index()
        labels = patient_labels.merge(sample_features[['patient_id']], on='patient_id', how='inner')
    else:
        print("Warning: No resistance_mechanism labels found. Using synthetic labels.")
        # Create synthetic labels for demonstration
        mechanisms = ['EMT', 'PI3K_AKT', 'EGFR_escape', 'efflux_pumps', 'apoptosis_escape', 'immune_escape']
        labels = pd.DataFrame({
            'patient_id': sample_features['patient_id'].unique(),
            'resistance_mechanism': np.random.choice(mechanisms, size=len(sample_features['patient_id'].unique()))
        })
    
    # Merge features and labels
    feature_df = sample_features.merge(labels, on='patient_id', how='inner')
    
    # Split features and labels
    feature_cols = [col for col in feature_df.columns if col not in ['patient_id', 'resistance_mechanism']]
    X = feature_df[feature_cols].fillna(0)
    y = feature_df[['patient_id', 'resistance_mechanism']]
    
    # Save features
    feature_file = output_dir / "sample_features.csv"
    feature_df.to_csv(feature_file, index=False)
    print(f"Saved features to {feature_file}")
    
    # 7. Model training and evaluation
    print("\n[7/7] Model training and evaluation...")
    
    if args.model == 'xgb':
        model = XGBResistancePredictor(random_state=args.seed)
        metrics = model.train(X, y, validation_split=0.2)
        
        # Predictions
        predictions = model.predict(X)
        predictions_df = pd.DataFrame(
            predictions,
            columns=[f'prob_{mech}' for mech in model.resistance_mechanisms]
        )
        predictions_df['patient_id'] = feature_df['patient_id'].values
        predictions_df['predicted_mechanism'] = [
            model.resistance_mechanisms[np.argmax(pred)]
            for pred in predictions
        ]
        
        # Save predictions
        pred_file = output_dir / "predictions.csv"
        predictions_df.to_csv(pred_file, index=False)
        print(f"Saved predictions to {pred_file}")
        
        # Feature importance
        importance_df = model.get_feature_importance()
        importance_file = output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        print(f"Saved feature importance to {importance_file}")
        
        # Save model
        model_file = output_dir / "xgb_model.pkl"
        model.save(str(model_file))
        print(f"Saved model to {model_file}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

