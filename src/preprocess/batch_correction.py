"""
Batch Correction Module for Single-Cell Data

Performs batch effect correction using various methods:
- Harmony
- Scanorama
- ComBat
- BBKNN
"""

import numpy as np
import scanpy as sc
import anndata as ad
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


def correct_batch_harmony(adata: ad.AnnData,
                         batch_key: str = 'patient_id',
                         n_components: int = 50,
                         use_rep: str = 'X_pca') -> ad.AnnData:
    """
    Correct batch effects using Harmony.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs containing batch information
    n_components : int
        Number of components to use
    use_rep : str
        Representation to use (should be PCA)
        
    Returns
    -------
    AnnData
        Batch-corrected AnnData
    """
    try:
        import harmonypy as hm
        
        # Ensure PCA is computed
        if use_rep not in adata.obsm:
            print("Computing PCA...")
            sc.tl.pca(adata, n_comps=n_components)
        
        # Run Harmony
        ho = hm.run_harmony(
            adata.obsm[use_rep],
            adata.obs,
            vars_use=[batch_key],
            max_iter_harmony=20
        )
        
        # Store corrected representation
        adata.obsm['X_pca_harmony'] = ho.Z_corr.T
        
        print("Harmony batch correction complete")
        
    except ImportError:
        print("Harmony not available. Install with: pip install harmonypy")
        print("Using PCA representation without correction")
        if use_rep not in adata.obsm:
            sc.tl.pca(adata, n_comps=n_components)
        adata.obsm['X_pca_harmony'] = adata.obsm[use_rep].copy()
    
    return adata


def correct_batch_scanorama(adata: ad.AnnData,
                            batch_key: str = 'patient_id',
                            n_components: int = 50) -> ad.AnnData:
    """
    Correct batch effects using Scanorama.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs containing batch information
    n_components : int
        Number of components to use
        
    Returns
    -------
    AnnData
        Batch-corrected AnnData
    """
    try:
        import scanorama
        
        # Split by batch
        batches = adata.obs[batch_key].unique()
        adatas = [adata[adata.obs[batch_key] == batch] for batch in batches]
        
        # Integrate
        integrated, corrected = scanorama.correct_scanpy(adatas, return_dimred=True)
        
        # Combine corrected data
        adata_corrected = ad.concat(integrated)
        adata.obsm['X_scanorama'] = adata_corrected.obsm['X_scanorama']
        
        print("Scanorama batch correction complete")
        
    except ImportError:
        print("Scanorama not available. Install with: pip install scanorama")
        print("Using PCA representation without correction")
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata, n_comps=n_components)
        adata.obsm['X_scanorama'] = adata.obsm['X_pca'].copy()
    
    return adata


def correct_batch_combat(adata: ad.AnnData,
                        batch_key: str = 'patient_id',
                        covariates: Optional[List[str]] = None) -> ad.AnnData:
    """
    Correct batch effects using ComBat.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs containing batch information
    covariates : list, optional
        Additional covariates to preserve
        
    Returns
    -------
    AnnData
        Batch-corrected AnnData
    """
    try:
        from combat.pycombat import pycombat
        
        # Use normalized data if available, otherwise use raw
        if 'log1p' in adata.layers:
            data = adata.layers['log1p'].T
        else:
            # Normalize if needed
            adata_normalized = adata.copy()
            sc.pp.normalize_total(adata_normalized, target_sum=1e4)
            sc.pp.log1p(adata_normalized)
            data = adata_normalized.X.T
        
        # Prepare batch and covariate information
        batch = adata.obs[batch_key].values
        
        if covariates is not None:
            covars = adata.obs[covariates].values
        else:
            covars = None
        
        # Run ComBat
        data_corrected = pycombat(data, batch, covars=covars)
        
        # Store corrected data
        adata.layers['combat_corrected'] = data_corrected.T
        
        print("ComBat batch correction complete")
        
    except ImportError:
        print("ComBat not available. Install with: pip install combat")
        print("Using normalized data without correction")
        if 'log1p' not in adata.layers:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        adata.layers['combat_corrected'] = adata.layers.get('log1p', adata.X).copy()
    
    return adata


def correct_batch_bbknn(adata: ad.AnnData,
                       batch_key: str = 'patient_id',
                       n_pcs: int = 50,
                       use_rep: str = 'X_pca') -> ad.AnnData:
    """
    Correct batch effects using BBKNN (Batch Balanced KNN).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    batch_key : str
        Key in adata.obs containing batch information
    n_pcs : int
        Number of principal components
    use_rep : str
        Representation to use
        
    Returns
    -------
    AnnData
        Batch-corrected AnnData with BBKNN graph
    """
    try:
        import bbknn
        
        # Ensure PCA is computed
        if use_rep not in adata.obsm:
            sc.tl.pca(adata, n_comps=n_pcs)
        
        # Run BBKNN
        bbknn.bbknn(adata, batch_key=batch_key, n_pcs=n_pcs)
        
        # Compute UMAP on BBKNN graph
        sc.tl.umap(adata)
        
        print("BBKNN batch correction complete")
        
    except ImportError:
        print("BBKNN not available. Install with: pip install bbknn")
        print("Computing standard neighbors graph")
        if use_rep not in adata.obsm:
            sc.tl.pca(adata, n_comps=n_pcs)
        sc.pp.neighbors(adata, n_pcs=n_pcs, use_rep=use_rep)
        sc.tl.umap(adata)
    
    return adata


def correct_batch(adata: ad.AnnData,
                  method: str = 'harmony',
                  batch_key: str = 'patient_id',
                  **kwargs) -> ad.AnnData:
    """
    Correct batch effects using specified method.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    method : str
        Method to use ('harmony', 'scanorama', 'combat', 'bbknn')
    batch_key : str
        Key in adata.obs containing batch information
    **kwargs
        Additional arguments for specific methods
        
    Returns
    -------
    AnnData
        Batch-corrected AnnData
    """
    method = method.lower()
    
    if method == 'harmony':
        return correct_batch_harmony(adata, batch_key=batch_key, **kwargs)
    elif method == 'scanorama':
        return correct_batch_scanorama(adata, batch_key=batch_key, **kwargs)
    elif method == 'combat':
        return correct_batch_combat(adata, batch_key=batch_key, **kwargs)
    elif method == 'bbknn':
        return correct_batch_bbknn(adata, batch_key=batch_key, **kwargs)
    else:
        raise ValueError(f"Unknown batch correction method: {method}")

