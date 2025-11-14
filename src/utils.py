"""
Utility Functions

Common utility functions for the pipeline.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


def load_adata(filepath: str) -> ad.AnnData:
    """
    Load AnnData object from file.
    
    Parameters
    ----------
    filepath : str
        Path to .h5ad file
        
    Returns
    -------
    AnnData
        Loaded AnnData object
    """
    return sc.read_h5ad(filepath)


def save_adata(adata: ad.AnnData, filepath: str):
    """
    Save AnnData object to file.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to save
    filepath : str
        Output filepath
    """
    adata.write(filepath)


def combine_adatas(adatas: List[ad.AnnData],
                  join: str = 'outer',
                  index_unique: str = '-') -> ad.AnnData:
    """
    Combine multiple AnnData objects.
    
    Parameters
    ----------
    adatas : list
        List of AnnData objects
    join : str
        Join method ('outer' or 'inner')
    index_unique : str
        Separator for unique indices
        
    Returns
    -------
    AnnData
        Combined AnnData object
    """
    return ad.concat(adatas, join=join, index_unique=index_unique)


def get_resistance_pathway_genes() -> Dict[str, List[str]]:
    """
    Get gene lists for resistance pathways.
    
    Returns
    -------
    dict
        Dictionary mapping pathway names to gene lists
    """
    # Example pathway genes (in real use, load from curated databases)
    pathways = {
        'EMT': [
            'VIM', 'FN1', 'CDH2', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2',
            'MMP2', 'MMP9', 'TGFB1', 'TGFBR1', 'TGFBR2'
        ],
        'PI3K_AKT': [
            'PIK3CA', 'PIK3CB', 'PIK3CD', 'AKT1', 'AKT2', 'AKT3', 'PTEN',
            'MTOR', 'RPS6KB1', 'EIF4EBP1', 'GSK3B'
        ],
        'EGFR': [
            'EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'KRAS', 'NRAS', 'BRAF',
            'MAPK1', 'MAPK3', 'JAK2', 'STAT3'
        ],
        'apoptosis_escape': [
            'BCL2', 'BCL2L1', 'MCL1', 'XIAP', 'CASP3', 'CASP7', 'CASP9',
            'TP53', 'MDM2', 'BAX', 'BAK1'
        ],
        'efflux_pumps': [
            'ABCB1', 'ABCC1', 'ABCC2', 'ABCG2', 'ABCB11'
        ],
        'immune_escape': [
            'PDL1', 'PDCD1LG2', 'CTLA4', 'LAG3', 'TIGIT', 'IDO1',
            'TGFB1', 'IL10', 'FOXP3'
        ]
    }
    
    return pathways


def normalize_data(adata: ad.AnnData,
                  method: str = 'log1p',
                  target_sum: float = 1e4) -> ad.AnnData:
    """
    Normalize single-cell data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    method : str
        Normalization method ('log1p' or 'sctransform')
    target_sum : float
        Target sum for normalization
        
    Returns
    -------
    AnnData
        Normalized AnnData
    """
    if method == 'log1p':
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X.copy()
    elif method == 'sctransform':
        try:
            import scanpy.external as sce
            sce.pp.scrublet(adata)  # Placeholder - would use SCTransform if available
            print("SCTransform normalization (placeholder)")
        except:
            print("SCTransform not available, using log1p")
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
            adata.layers['log1p'] = adata.X.copy()
    
    return adata


def find_hvg(adata: ad.AnnData,
            n_top_genes: int = 2000,
            flavor: str = 'seurat') -> ad.AnnData:
    """
    Find highly variable genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    n_top_genes : int
        Number of top variable genes
    flavor : str
        Method flavor
        
    Returns
    -------
    AnnData
        AnnData with HVG information
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    return adata

