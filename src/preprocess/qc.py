"""
Quality Control Module for Single-Cell Data

Performs quality control including:
- Mitochondrial gene percentage
- Number of features (genes) per cell
- Doublet detection
- Cell filtering
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def calculate_qc_metrics(adata: ad.AnnData, 
                        mito_prefix: str = 'MT-',
                        ribo_prefix: str = 'RPS|RPL') -> ad.AnnData:
    """
    Calculate quality control metrics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    mito_prefix : str
        Prefix for mitochondrial genes
    ribo_prefix : str
        Prefix for ribosomal genes
        
    Returns
    -------
    AnnData
        AnnData with QC metrics added to obs
    """
    adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)
    adata.var['ribo'] = adata.var_names.str.contains(ribo_prefix, case=False, regex=True)
    
    # Calculate QC metrics
    adata.obs['n_genes_by_counts'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['pct_counts_mt'] = np.array(
        adata[:, adata.var['mt']].X.sum(axis=1) / adata.obs['total_counts']
    ).flatten() * 100
    adata.obs['pct_counts_ribo'] = np.array(
        adata[:, adata.var['ribo']].X.sum(axis=1) / adata.obs['total_counts']
    ).flatten() * 100
    
    return adata


def detect_doublets(adata: ad.AnnData,
                   method: str = 'scrublet',
                   expected_doublet_rate: float = 0.1,
                   n_neighbors: int = 30) -> ad.AnnData:
    """
    Detect doublets using Scrublet or simple heuristic.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    method : str
        Method to use ('scrublet' or 'heuristic')
    expected_doublet_rate : float
        Expected doublet rate
    n_neighbors : int
        Number of neighbors for Scrublet
        
    Returns
    -------
    AnnData
        AnnData with doublet scores and predictions
    """
    if method == 'scrublet':
        try:
            import scrublet as scr
            scrub = scr.Scrublet(adata.X, expected_doublet_rate=expected_doublet_rate)
            doublet_scores, predicted_doublets = scrub.scrub_doublets(
                min_counts=2,
                min_cells=3,
                n_prin_comps=30
            )
            adata.obs['doublet_score'] = doublet_scores
            adata.obs['predicted_doublet'] = predicted_doublets
        except ImportError:
            print("Scrublet not available, using heuristic method")
            method = 'heuristic'
    
    if method == 'heuristic':
        # Simple heuristic: cells with very high gene counts
        n_genes_threshold = np.percentile(adata.obs['n_genes_by_counts'], 95)
        total_counts_threshold = np.percentile(adata.obs['total_counts'], 95)
        
        predicted_doublets = (
            (adata.obs['n_genes_by_counts'] > n_genes_threshold) &
            (adata.obs['total_counts'] > total_counts_threshold)
        )
        
        # Calculate doublet score as normalized distance from median
        median_genes = adata.obs['n_genes_by_counts'].median()
        median_counts = adata.obs['total_counts'].median()
        
        gene_score = (adata.obs['n_genes_by_counts'] - median_genes) / median_genes
        count_score = (adata.obs['total_counts'] - median_counts) / median_counts
        doublet_scores = (gene_score + count_score) / 2
        
        adata.obs['doublet_score'] = doublet_scores
        adata.obs['predicted_doublet'] = predicted_doublets
    
    return adata


def filter_cells(adata: ad.AnnData,
                min_genes: Optional[int] = None,
                max_genes: Optional[int] = None,
                min_counts: Optional[int] = None,
                max_counts: Optional[int] = None,
                max_pct_mt: float = 20.0,
                remove_doublets: bool = True) -> Tuple[ad.AnnData, Dict]:
    """
    Filter cells based on QC metrics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    min_genes : int, optional
        Minimum number of genes
    max_genes : int, optional
        Maximum number of genes
    min_counts : int, optional
        Minimum total counts
    max_counts : int, optional
        Maximum total counts
    max_pct_mt : float
        Maximum percentage of mitochondrial counts
    remove_doublets : bool
        Whether to remove predicted doublets
        
    Returns
    -------
    Tuple[AnnData, Dict]
        Filtered AnnData and filtering statistics
    """
    n_cells_before = adata.n_obs
    
    # Set defaults based on data if not provided
    if min_genes is None:
        min_genes = np.percentile(adata.obs['n_genes_by_counts'], 5)
    if max_genes is None:
        max_genes = np.percentile(adata.obs['n_genes_by_counts'], 99)
    if min_counts is None:
        min_counts = np.percentile(adata.obs['total_counts'], 5)
    if max_counts is None:
        max_counts = np.percentile(adata.obs['total_counts'], 99)
    
    # Create filter mask
    filter_mask = np.ones(adata.n_obs, dtype=bool)
    
    # Filter by gene count
    filter_mask &= (adata.obs['n_genes_by_counts'] >= min_genes)
    filter_mask &= (adata.obs['n_genes_by_counts'] <= max_genes)
    
    # Filter by total counts
    filter_mask &= (adata.obs['total_counts'] >= min_counts)
    filter_mask &= (adata.obs['total_counts'] <= max_counts)
    
    # Filter by mitochondrial percentage
    filter_mask &= (adata.obs['pct_counts_mt'] <= max_pct_mt)
    
    # Remove doublets
    if remove_doublets and 'predicted_doublet' in adata.obs:
        filter_mask &= ~adata.obs['predicted_doublet']
    
    # Apply filter
    adata_filtered = adata[filter_mask].copy()
    
    n_cells_after = adata_filtered.n_obs
    n_removed = n_cells_before - n_cells_after
    
    stats = {
        'n_cells_before': n_cells_before,
        'n_cells_after': n_cells_after,
        'n_removed': n_removed,
        'pct_removed': (n_removed / n_cells_before) * 100,
        'filter_criteria': {
            'min_genes': min_genes,
            'max_genes': max_genes,
            'min_counts': min_counts,
            'max_counts': max_counts,
            'max_pct_mt': max_pct_mt,
            'remove_doublets': remove_doublets
        }
    }
    
    return adata_filtered, stats


def filter_genes(adata: ad.AnnData,
                min_cells: int = 3) -> ad.AnnData:
    """
    Filter genes expressed in too few cells.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    min_cells : int
        Minimum number of cells expressing the gene
        
    Returns
    -------
    AnnData
        Filtered AnnData
    """
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def run_qc(adata: ad.AnnData,
          min_genes: Optional[int] = None,
          max_genes: Optional[int] = None,
          min_counts: Optional[int] = None,
          max_counts: Optional[int] = None,
          max_pct_mt: float = 20.0,
          min_cells: int = 3,
          remove_doublets: bool = True,
          doublet_method: str = 'heuristic') -> Tuple[ad.AnnData, Dict]:
    """
    Run complete QC pipeline.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    min_genes : int, optional
        Minimum number of genes per cell
    max_genes : int, optional
        Maximum number of genes per cell
    min_counts : int, optional
        Minimum total counts per cell
    max_counts : int, optional
        Maximum total counts per cell
    max_pct_mt : float
        Maximum percentage of mitochondrial counts
    min_cells : int
        Minimum number of cells expressing a gene
    remove_doublets : bool
        Whether to remove doublets
    doublet_method : str
        Method for doublet detection
        
    Returns
    -------
    Tuple[AnnData, Dict]
        QC'd AnnData and statistics
    """
    print("Calculating QC metrics...")
    adata = calculate_qc_metrics(adata)
    
    print("Detecting doublets...")
    adata = detect_doublets(adata, method=doublet_method)
    
    print("Filtering cells...")
    adata, cell_stats = filter_cells(
        adata,
        min_genes=min_genes,
        max_genes=max_genes,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        remove_doublets=remove_doublets
    )
    
    print("Filtering genes...")
    n_genes_before = adata.n_vars
    adata = filter_genes(adata, min_cells=min_cells)
    n_genes_after = adata.n_vars
    
    stats = {
        **cell_stats,
        'n_genes_before': n_genes_before,
        'n_genes_after': n_genes_after,
        'n_genes_removed': n_genes_before - n_genes_after
    }
    
    print(f"QC complete: {stats['n_cells_after']:,} cells, {stats['n_genes_after']:,} genes")
    
    return adata, stats

