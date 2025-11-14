"""
Graph Neural Network Model for Resistance Prediction

Uses GNN to model cell-cell relationships and predict resistance.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class GNNResistancePredictor(nn.Module):
    """Graph Neural Network for resistance prediction."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_classes: int = 6,
                 n_layers: int = 3,
                 dropout: float = 0.3,
                 gnn_type: str = 'GCN'):
        """
        Initialize GNN model.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dim : int
            Hidden layer dimension
        n_classes : int
            Number of output classes
        n_layers : int
            Number of GNN layers
        dropout : float
            Dropout rate
        gnn_type : str
            GNN type ('GCN' or 'GAT')
        """
        super(GNNResistancePredictor, self).__init__()
        
        self.gnn_type = gnn_type
        self.n_layers = n_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(input_dim, hidden_dim) if gnn_type == 'GCN' 
            else GATConv(input_dim, hidden_dim, heads=4, concat=False)
        )
        
        for _ in range(n_layers - 1):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim) if gnn_type == 'GCN'
                else GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            )
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices
        batch : torch.Tensor, optional
            Batch assignment
            
        Returns
        -------
        torch.Tensor
            Graph-level predictions
        """
        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        # Classify
        x = self.classifier(x)
        
        return x


def build_cell_graph(cell_features: np.ndarray,
                    k: int = 15,
                    distance_metric: str = 'euclidean') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-nearest neighbor graph from cell features.
    
    Parameters
    ----------
    cell_features : np.ndarray
        Cell feature matrix
    k : int
        Number of neighbors
    distance_metric : str
        Distance metric
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Edge indices and edge weights
    """
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=distance_metric).fit(cell_features)
    distances, indices = nbrs.kneighbors(cell_features)
    
    # Remove self-connections
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    
    # Create edge list
    n_cells = cell_features.shape[0]
    edge_list = []
    edge_weights = []
    
    for i in range(n_cells):
        for j, neighbor_idx in enumerate(indices[i]):
            edge_list.append([i, neighbor_idx])
            edge_weights.append(1.0 / (distances[i, j] + 1e-6))
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_weights

