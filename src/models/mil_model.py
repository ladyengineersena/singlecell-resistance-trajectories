"""
Multiple Instance Learning (MIL) Model for Resistance Prediction

Uses attention-based pooling to aggregate cell-level features into sample-level predictions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MILDataset(Dataset):
    """Dataset for Multiple Instance Learning."""
    
    def __init__(self, cell_features: Dict[str, np.ndarray], labels: pd.DataFrame):
        """
        Initialize MIL dataset.
        
        Parameters
        ----------
        cell_features : dict
            Dictionary mapping patient_id to cell feature matrix
        labels : pd.DataFrame
            Labels with patient_id column
        """
        self.patient_ids = labels['patient_id'].values
        self.cell_features = cell_features
        self.labels = labels.drop('patient_id', axis=1).values if 'patient_id' in labels.columns else labels.values
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        features = torch.FloatTensor(self.cell_features[patient_id])
        label = torch.FloatTensor(self.labels[idx])
        return features, label, patient_id


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning model."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_classes: int = 6,
                 dropout: float = 0.3):
        """
        Initialize Attention MIL model.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dim : int
            Hidden layer dimension
        n_classes : int
            Number of output classes
        dropout : float
            Dropout rate
        """
        super(AttentionMIL, self).__init__()
        
        # Feature transformation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch_size, n_cells, input_dim)
            
        Returns
        -------
        torch.Tensor
            Predictions (batch_size, n_classes)
        torch.Tensor
            Attention weights (batch_size, n_cells)
        """
        # Extract features
        H = self.feature_extractor(x)  # (batch, n_cells, hidden_dim)
        
        # Compute attention weights
        A = self.attention(H)  # (batch, n_cells, 1)
        A = torch.transpose(A, 1, 0)  # (n_cells, batch, 1)
        A = torch.transpose(A, 1, 2)  # (n_cells, 1, batch)
        A = torch.softmax(A, dim=0)  # (n_cells, 1, batch)
        A = torch.transpose(A, 1, 2)  # (n_cells, batch, 1)
        A = torch.transpose(A, 0, 1)  # (batch, n_cells, 1)
        
        # Aggregate with attention
        M = torch.sum(A * H, dim=1)  # (batch, hidden_dim)
        
        # Classify
        Y = self.classifier(M)  # (batch, n_classes)
        
        return Y, A.squeeze(-1)


class MILResistancePredictor:
    """MIL-based predictor for treatment resistance."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_classes: int = 6,
                 learning_rate: float = 0.001,
                 batch_size: int = 8,
                 n_epochs: int = 50,
                 device: str = 'cpu'):
        """
        Initialize MIL predictor.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dim : int
            Hidden layer dimension
        n_classes : int
            Number of output classes
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size
        n_epochs : int
            Number of training epochs
        device : str
            Device ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = AttentionMIL(input_dim, hidden_dim, n_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def _prepare_cell_features(self, cell_features: np.ndarray, metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare cell features grouped by patient.
        
        Parameters
        ----------
        cell_features : np.ndarray
            Cell feature matrix
        metadata : pd.DataFrame
            Cell metadata with patient_id
            
        Returns
        -------
        dict
            Dictionary mapping patient_id to cell features
        """
        if 'patient_id' not in metadata.columns:
            raise ValueError("metadata must contain 'patient_id' column")
        
        patient_features = {}
        for patient_id in metadata['patient_id'].unique():
            patient_mask = metadata['patient_id'] == patient_id
            patient_features[patient_id] = cell_features[patient_mask]
        
        return patient_features
    
    def train(self,
             cell_features: np.ndarray,
             cell_metadata: pd.DataFrame,
             labels: pd.DataFrame,
             validation_split: float = 0.2) -> Dict:
        """
        Train MIL model.
        
        Parameters
        ----------
        cell_features : np.ndarray
            Cell feature matrix
        cell_metadata : pd.DataFrame
            Cell metadata
        labels : pd.DataFrame
            Sample-level labels
        validation_split : float
            Validation split fraction
            
        Returns
        -------
        dict
            Training history
        """
        # Prepare cell features by patient
        patient_features = self._prepare_cell_features(cell_features, cell_metadata)
        
        # Split patients
        patient_ids = labels['patient_id'].values
        n_train = int(len(patient_ids) * (1 - validation_split))
        indices = np.random.permutation(len(patient_ids))
        train_ids = patient_ids[indices[:n_train]]
        val_ids = patient_ids[indices[n_train:]]
        
        train_labels = labels[labels['patient_id'].isin(train_ids)]
        val_labels = labels[labels['patient_id'].isin(val_ids)]
        
        # Create datasets
        train_dataset = MILDataset(
            {pid: patient_features[pid] for pid in train_ids},
            train_labels
        )
        val_dataset = MILDataset(
            {pid: patient_features[pid] for pid in val_ids},
            val_labels
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for features, labels_batch, _ in train_loader:
                features = features.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                # Pad sequences to same length
                max_len = max(f.shape[0] for f in features)
                padded_features = []
                for f in features:
                    pad_size = max_len - f.shape[0]
                    if pad_size > 0:
                        f = torch.cat([f, torch.zeros(pad_size, f.shape[1])], dim=0)
                    padded_features.append(f)
                features = torch.stack(padded_features)
                
                self.optimizer.zero_grad()
                predictions, _ = self.model(features)
                loss = self.criterion(predictions, labels_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_labels_list = []
            
            with torch.no_grad():
                for features, labels_batch, _ in val_loader:
                    features = features.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    # Pad sequences
                    max_len = max(f.shape[0] for f in features)
                    padded_features = []
                    for f in features:
                        pad_size = max_len - f.shape[0]
                        if pad_size > 0:
                            f = torch.cat([f, torch.zeros(pad_size, f.shape[1])], dim=0)
                        padded_features.append(f)
                    features = torch.stack(padded_features)
                    
                    predictions, _ = self.model(features)
                    loss = self.criterion(predictions, labels_batch)
                    val_loss += loss.item()
                    
                    val_predictions.append(predictions.cpu().numpy())
                    val_labels_list.append(labels_batch.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_predictions = np.concatenate(val_predictions, axis=0)
            val_labels_list = np.concatenate(val_labels_list, axis=0)
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(val_labels_list, val_predictions, average='macro')
            except:
                val_auc = 0.0
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        return history
    
    def predict(self,
               cell_features: np.ndarray,
               cell_metadata: pd.DataFrame) -> np.ndarray:
        """
        Predict resistance mechanisms.
        
        Parameters
        ----------
        cell_features : np.ndarray
            Cell feature matrix
        cell_metadata : pd.DataFrame
            Cell metadata
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()
        patient_features = self._prepare_cell_features(cell_features, cell_metadata)
        
        predictions = {}
        attention_weights = {}
        
        with torch.no_grad():
            for patient_id, features in patient_features.items():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                pred, att = self.model(features_tensor)
                predictions[patient_id] = pred.cpu().numpy()[0]
                attention_weights[patient_id] = att.cpu().numpy()[0]
        
        return predictions, attention_weights

