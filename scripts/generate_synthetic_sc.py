"""
Synthetic Single-Cell Data Generator for Treatment Resistance Trajectories

This script generates synthetic single-cell datasets that simulate:
- Multi-timepoint longitudinal samples
- Multiple patients with different resistance trajectories
- Various resistance mechanisms (EMT, PI3K/AKT, EGFR escape, etc.)
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Resistance mechanisms
RESISTANCE_MECHANISMS = [
    'EMT',              # Epithelial-mesenchymal transition
    'PI3K_AKT',         # PI3K/AKT pathway activation
    'EGFR_escape',      # EGFR mutation escape
    'efflux_pumps',     # Drug efflux pumps
    'apoptosis_escape', # Apoptosis resistance
    'immune_escape'     # Immune evasion
]

# Timepoints
TIMEPOINTS = ['pre_treatment', 'on_treatment_early', 'on_treatment_late', 'progression']


def generate_gene_expression(
    n_cells: int,
    n_genes: int = 2000,
    base_expression: float = 5.0,
    noise_level: float = 1.0,
    mechanism: str = None,
    timepoint: str = 'pre_treatment',
    mechanism_strength: float = 2.0
) -> np.ndarray:
    """
    Generate synthetic gene expression matrix.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    base_expression : float
        Base expression level
    noise_level : float
        Noise level
    mechanism : str
        Resistance mechanism (affects specific gene sets)
    timepoint : str
        Timepoint (affects expression levels)
    mechanism_strength : float
        Strength of mechanism signal
        
    Returns
    -------
    np.ndarray
        Expression matrix (cells × genes)
    """
    # Base expression with noise
    expression = np.random.negative_binomial(
        n=base_expression,
        p=0.3,
        size=(n_cells, n_genes)
    ).astype(float)
    
    # Add mechanism-specific signals
    if mechanism:
        mechanism_genes = get_mechanism_genes(mechanism, n_genes)
        timepoint_multiplier = get_timepoint_multiplier(timepoint)
        
        for gene_idx in mechanism_genes:
            # Increase expression for mechanism genes at later timepoints
            expression[:, gene_idx] *= (1 + mechanism_strength * timepoint_multiplier * 
                                       np.random.uniform(0.5, 1.5, n_cells))
    
    # Add noise
    expression += np.random.normal(0, noise_level, expression.shape)
    expression = np.maximum(expression, 0)  # Ensure non-negative
    
    return expression


def get_mechanism_genes(mechanism: str, n_genes: int) -> List[int]:
    """Get gene indices associated with a resistance mechanism."""
    # Simulate mechanism-specific gene sets
    np.random.seed(hash(mechanism) % 2**32)
    n_mechanism_genes = min(50, n_genes // 10)
    return np.random.choice(n_genes, size=n_mechanism_genes, replace=False).tolist()


def get_timepoint_multiplier(timepoint: str) -> float:
    """Get timepoint-specific multiplier for expression changes."""
    multipliers = {
        'pre_treatment': 0.0,
        'on_treatment_early': 0.3,
        'on_treatment_late': 0.7,
        'progression': 1.0
    }
    return multipliers.get(timepoint, 0.0)


def generate_spliced_unspliced(
    expression: np.ndarray,
    velocity_scale: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spliced and unspliced counts for RNA velocity.
    
    Parameters
    ----------
    expression : np.ndarray
        Total expression matrix
    velocity_scale : float
        Scale for velocity dynamics
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Spliced and unspliced matrices
    """
    # Simulate splicing dynamics
    splicing_ratio = np.random.beta(2, 1, size=expression.shape)
    spliced = (expression * splicing_ratio * (1 + velocity_scale)).astype(int)
    unspliced = (expression * (1 - splicing_ratio) * (1 - velocity_scale)).astype(int)
    
    # Ensure non-negative and reasonable values
    spliced = np.maximum(spliced, 0)
    unspliced = np.maximum(unspliced, 0)
    
    return spliced, unspliced


def create_patient_data(
    patient_id: str,
    n_cells_per_timepoint: int = 2000,
    n_timepoints: int = 3,
    resistance_mechanism: str = None,
    treatment_type: str = 'targeted_therapy',
    responder: bool = True
) -> List[ad.AnnData]:
    """
    Create synthetic data for a single patient across timepoints.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    n_cells_per_timepoint : int
        Number of cells per timepoint
    n_timepoints : int
        Number of timepoints
    resistance_mechanism : str
        Primary resistance mechanism
    treatment_type : str
        Treatment type
    responder : bool
        Whether patient responds to treatment
        
    Returns
    -------
    List[ad.AnnData]
        List of AnnData objects, one per timepoint
    """
    if resistance_mechanism is None:
        resistance_mechanism = np.random.choice(RESISTANCE_MECHANISMS)
    
    n_genes = 2000
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    adatas = []
    selected_timepoints = TIMEPOINTS[:n_timepoints]
    
    for tp_idx, timepoint in enumerate(selected_timepoints):
        # Adjust mechanism strength based on timepoint and response
        if responder and timepoint == 'progression':
            # Non-responders develop resistance
            mechanism_strength = 3.0
        elif not responder and timepoint in ['on_treatment_early', 'on_treatment_late']:
            # Early resistance in non-responders
            mechanism_strength = 2.0
        else:
            mechanism_strength = 0.5
        
        # Generate expression
        expression = generate_gene_expression(
            n_cells=n_cells_per_timepoint,
            n_genes=n_genes,
            mechanism=resistance_mechanism if tp_idx > 0 else None,
            timepoint=timepoint,
            mechanism_strength=mechanism_strength
        )
        
        # Generate spliced/unspliced for velocity
        spliced, unspliced = generate_spliced_unspliced(expression)
        
        # Create cell metadata
        cell_metadata = pd.DataFrame({
            'patient_id': [patient_id] * n_cells_per_timepoint,
            'timepoint': [timepoint] * n_cells_per_timepoint,
            'treatment_type': [treatment_type] * n_cells_per_timepoint,
            'resistance_mechanism': [resistance_mechanism] * n_cells_per_timepoint,
            'responder': [responder] * n_cells_per_timepoint,
            'days_since_treatment': [tp_idx * 30] * n_cells_per_timepoint,  # Relative days
            'cell_type': np.random.choice(
                ['T_cell', 'B_cell', 'Tumor', 'Stromal', 'Endothelial'],
                size=n_cells_per_timepoint,
                p=[0.2, 0.1, 0.5, 0.15, 0.05]
            )
        })
        
        # Create AnnData object
        adata = ad.AnnData(
            X=expression.astype(int),
            obs=cell_metadata,
            var=pd.DataFrame(index=gene_names)
        )
        
        # Add layers for velocity
        adata.layers['spliced'] = spliced
        adata.layers['unspliced'] = unspliced
        adata.layers['counts'] = expression.astype(int)
        
        adatas.append(adata)
    
    return adatas


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic single-cell data for resistance trajectories'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/synthetic',
        help='Output directory'
    )
    parser.add_argument(
        '--n_patients',
        type=int,
        default=20,
        help='Number of patients'
    )
    parser.add_argument(
        '--n_cells_per_patient',
        type=int,
        default=2000,
        help='Number of cells per timepoint per patient'
    )
    parser.add_argument(
        '--n_timepoints',
        type=int,
        default=3,
        help='Number of timepoints per patient'
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic data for {args.n_patients} patients...")
    print(f"Output directory: {out_dir}")
    
    all_adatas = []
    patient_metadata = []
    
    for patient_idx in range(args.n_patients):
        patient_id = f"Patient_{patient_idx:03d}"
        
        # Randomly assign resistance mechanism and response status
        resistance_mechanism = np.random.choice(RESISTANCE_MECHANISMS)
        responder = np.random.choice([True, False], p=[0.6, 0.4])
        treatment_type = np.random.choice(['targeted_therapy', 'immunotherapy', 'chemotherapy'])
        
        # Generate data for this patient
        patient_adatas = create_patient_data(
            patient_id=patient_id,
            n_cells_per_timepoint=args.n_cells_per_patient,
            n_timepoints=args.n_timepoints,
            resistance_mechanism=resistance_mechanism,
            treatment_type=treatment_type,
            responder=responder
        )
        
        # Save individual timepoint files
        for adata in patient_adatas:
            timepoint = adata.obs['timepoint'].iloc[0]
            filename = out_dir / f"{patient_id}_{timepoint}.h5ad"
            adata.write(filename)
            all_adatas.append(adata)
        
        # Store patient-level metadata
        patient_metadata.append({
            'patient_id': patient_id,
            'resistance_mechanism': resistance_mechanism,
            'responder': responder,
            'treatment_type': treatment_type,
            'n_timepoints': args.n_timepoints
        })
    
    # Combine all data
    print("Combining all data...")
    combined_adata = ad.concat(all_adatas, join='outer', index_unique='-')
    
    # Save combined file
    combined_file = out_dir / "synthetic_combined.h5ad"
    combined_adata.write(combined_file)
    print(f"Saved combined data to {combined_file}")
    
    # Save patient metadata
    patient_df = pd.DataFrame(patient_metadata)
    metadata_file = out_dir / "patient_metadata.csv"
    patient_df.to_csv(metadata_file, index=False)
    print(f"Saved patient metadata to {metadata_file}")
    
    print(f"\nGenerated data summary:")
    print(f"  Total cells: {combined_adata.n_obs:,}")
    print(f"  Total genes: {combined_adata.n_vars:,}")
    print(f"  Patients: {args.n_patients}")
    print(f"  Timepoints per patient: {args.n_timepoints}")
    print(f"\nDone! Data saved to {out_dir}")


if __name__ == '__main__':
    main()

