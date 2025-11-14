"""
Feature Engineering Module

Extracts features at multiple levels:
- Cell-level embeddings
- Cluster-level dynamics
- Regulatory features
- Trajectory features
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureBuilder:
    """Build features from single-cell data for resistance prediction."""
    
    def __init__(self, adata: ad.AnnData):
        """
        Initialize FeatureBuilder.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object with trajectory information
        """
        self.adata = adata
        self.features = {}
    
    def extract_cell_embeddings(self, 
                                use_rep: str = 'X_pca',
                                n_components: int = 50) -> np.ndarray:
        """
        Extract cell-level embeddings.
        
        Parameters
        ----------
        use_rep : str
            Representation to use
        n_components : int
            Number of components to extract
            
        Returns
        -------
        np.ndarray
            Cell embeddings (n_cells × n_components)
        """
        if use_rep in self.adata.obsm:
            embeddings = self.adata.obsm[use_rep]
            if embeddings.shape[1] > n_components:
                embeddings = embeddings[:, :n_components]
            return embeddings
        else:
            # Fallback: use raw expression (reduced)
            if self.adata.n_vars > n_components:
                pca = PCA(n_components=n_components)
                embeddings = pca.fit_transform(self.adata.X)
            else:
                embeddings = self.adata.X
            return embeddings
    
    def extract_cluster_features(self,
                                 cluster_key: str = 'leiden',
                                 timepoint_key: str = 'timepoint') -> pd.DataFrame:
        """
        Extract cluster-level dynamic features.
        
        Parameters
        ----------
        cluster_key : str
            Key in adata.obs for cluster labels
        timepoint_key : str
            Key in adata.obs for timepoint information
            
        Returns
        -------
        pd.DataFrame
            Cluster features per sample
        """
        if cluster_key not in self.adata.obs:
            # Perform clustering if not done
            print(f"Clustering not found. Computing {cluster_key}...")
            sc.tl.leiden(self.adata, key_added=cluster_key)
        
        features_list = []
        
        # Group by patient and timepoint
        if 'patient_id' in self.adata.obs and timepoint_key in self.adata.obs:
            for (patient_id, timepoint), group in self.adata.obs.groupby(['patient_id', timepoint_key]):
                cluster_counts = group[cluster_key].value_counts()
                total_cells = len(group)
                
                # Cluster frequencies
                cluster_freq = (cluster_counts / total_cells).to_dict()
                
                # Number of clusters
                n_clusters = len(cluster_counts)
                
                # Shannon diversity
                shannon = -sum((freq * np.log(freq + 1e-10) for freq in cluster_freq.values()))
                
                features = {
                    'patient_id': patient_id,
                    'timepoint': timepoint,
                    'n_clusters': n_clusters,
                    'shannon_diversity': shannon,
                    **{f'cluster_{k}_freq': v for k, v in cluster_freq.items()}
                }
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_trajectory_features(self,
                                   pseudotime_key: str = 'velocity_pseudotime',
                                   timepoint_key: str = 'timepoint') -> pd.DataFrame:
        """
        Extract trajectory-based features.
        
        Parameters
        ----------
        pseudotime_key : str
            Key in adata.obs for pseudotime
        timepoint_key : str
            Key in adata.obs for timepoint
            
        Returns
        -------
        pd.DataFrame
            Trajectory features per sample
        """
        features_list = []
        
        if 'patient_id' in self.adata.obs:
            for patient_id, group_idx in self.adata.obs.groupby('patient_id').groups.items():
                patient_mask = self.adata.obs['patient_id'] == patient_id
                patient_data = self.adata[patient_mask]
                
                # Pseudotime statistics
                if pseudotime_key in patient_data.obs:
                    pseudotime = patient_data.obs[pseudotime_key].values
                    features = {
                        'patient_id': patient_id,
                        'pseudotime_mean': np.mean(pseudotime),
                        'pseudotime_std': np.std(pseudotime),
                        'pseudotime_max': np.max(pseudotime),
                        'pseudotime_min': np.min(pseudotime),
                        'pseudotime_range': np.max(pseudotime) - np.min(pseudotime)
                    }
                else:
                    features = {'patient_id': patient_id}
                
                # Timepoint progression
                if timepoint_key in patient_data.obs:
                    timepoints = patient_data.obs[timepoint_key].unique()
                    timepoint_order = {
                        'pre_treatment': 0,
                        'on_treatment_early': 1,
                        'on_treatment_late': 2,
                        'progression': 3
                    }
                    max_timepoint = max([timepoint_order.get(tp, 0) for tp in timepoints])
                    features['max_timepoint_ordinal'] = max_timepoint
                    features['n_timepoints'] = len(timepoints)
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_pathway_features(self,
                                pathway_genes: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Extract pathway activity features.
        
        Parameters
        ----------
        pathway_genes : dict
            Dictionary mapping pathway names to gene lists
            
        Returns
        -------
        pd.DataFrame
            Pathway features per sample
        """
        features_list = []
        
        # Use normalized data if available
        if 'log1p' in self.adata.layers:
            expression = self.adata.layers['log1p']
        else:
            expression = self.adata.X
        
        # Compute pathway scores (mean expression of pathway genes)
        pathway_scores = {}
        for pathway_name, genes in pathway_genes.items():
            # Find matching genes
            gene_mask = self.adata.var_names.isin(genes)
            if gene_mask.sum() > 0:
                pathway_expression = expression[:, gene_mask].mean(axis=1)
                pathway_scores[pathway_name] = np.array(pathway_expression).flatten()
            else:
                pathway_scores[pathway_name] = np.zeros(self.adata.n_obs)
        
        # Aggregate by patient
        if 'patient_id' in self.adata.obs:
            for patient_id, group_idx in self.adata.obs.groupby('patient_id').groups.items():
                patient_mask = self.adata.obs['patient_id'] == patient_id
                features = {'patient_id': patient_id}
                
                for pathway_name, scores in pathway_scores.items():
                    patient_scores = scores[patient_mask]
                    features[f'{pathway_name}_mean'] = np.mean(patient_scores)
                    features[f'{pathway_name}_std'] = np.std(patient_scores)
                    features[f'{pathway_name}_max'] = np.max(patient_scores)
                
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_sample_features(self,
                               include_embeddings: bool = True,
                               include_clusters: bool = True,
                               include_trajectory: bool = True,
                               include_pathways: bool = False,
                               pathway_genes: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Extract all sample-level features.
        
        Parameters
        ----------
        include_embeddings : bool
            Include cell embedding aggregates
        include_clusters : bool
            Include cluster features
        include_trajectory : bool
            Include trajectory features
        include_pathways : bool
            Include pathway features
        pathway_genes : dict, optional
            Pathway gene definitions
            
        Returns
        -------
        pd.DataFrame
            Combined features per sample
        """
        feature_dfs = []
        
        # Patient ID as base
        if 'patient_id' in self.adata.obs:
            patient_ids = self.adata.obs['patient_id'].unique()
            base_df = pd.DataFrame({'patient_id': patient_ids})
            feature_dfs.append(base_df)
        
        # Cluster features
        if include_clusters:
            cluster_features = self.extract_cluster_features()
            if not cluster_features.empty:
                feature_dfs.append(cluster_features)
        
        # Trajectory features
        if include_trajectory:
            trajectory_features = self.extract_trajectory_features()
            if not trajectory_features.empty:
                feature_dfs.append(trajectory_features)
        
        # Pathway features
        if include_pathways and pathway_genes:
            pathway_features = self.extract_pathway_features(pathway_genes)
            if not pathway_features.empty:
                feature_dfs.append(pathway_features)
        
        # Merge all features
        if feature_dfs:
            combined = feature_dfs[0]
            for df in feature_dfs[1:]:
                if 'patient_id' in df.columns:
                    combined = combined.merge(df, on='patient_id', how='outer')
            return combined
        else:
            return pd.DataFrame()
    
    def get_cell_level_features(self,
                               use_rep: str = 'X_pca',
                               n_components: int = 50) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get cell-level features for MIL models.
        
        Parameters
        ----------
        use_rep : str
            Representation to use
        n_components : int
            Number of components
            
        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            Cell embeddings and metadata
        """
        embeddings = self.extract_cell_embeddings(use_rep=use_rep, n_components=n_components)
        metadata = self.adata.obs.copy()
        
        return embeddings, metadata

