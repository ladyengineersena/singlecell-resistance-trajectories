"""
XGBoost Model for Resistance Prediction

Provides XGBoost-based classifier for multi-label resistance mechanism prediction.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_fscore_support, classification_report
)
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')


class XGBResistancePredictor:
    """XGBoost-based predictor for treatment resistance mechanisms."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 objective: str = 'binary:logistic',
                 random_state: int = 42):
        """
        Initialize XGBoost predictor.
        
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        objective : str
            Objective function
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        self.resistance_mechanisms = None
    
    def _prepare_labels(self, labels: pd.DataFrame) -> np.ndarray:
        """
        Prepare multi-label targets.
        
        Parameters
        ----------
        labels : pd.DataFrame
            DataFrame with resistance mechanism columns
            
        Returns
        -------
        np.ndarray
            Binary matrix (n_samples × n_mechanisms)
        """
        # If labels is a single column with mechanism names, convert to multi-label
        if labels.shape[1] == 1:
            mechanism_col = labels.columns[0]
            mechanisms = labels[mechanism_col].unique()
            self.resistance_mechanisms = sorted(mechanisms)
            
            # Create binary matrix
            label_matrix = np.zeros((len(labels), len(self.resistance_mechanisms)))
            for i, mechanism in enumerate(self.resistance_mechanisms):
                label_matrix[:, i] = (labels[mechanism_col] == mechanism).astype(int)
            
            return label_matrix
        else:
            # Assume already in multi-label format
            self.resistance_mechanisms = labels.columns.tolist()
            return labels.values
    
    def train(self,
             X: pd.DataFrame,
             y: pd.DataFrame,
             validation_split: float = 0.2,
             early_stopping_rounds: int = 10) -> Dict:
        """
        Train XGBoost model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.DataFrame
            Labels (can be single column or multi-label)
        validation_split : float
            Fraction for validation
        early_stopping_rounds : int
            Early stopping rounds
            
        Returns
        -------
        dict
            Training history and metrics
        """
        # Prepare features
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)
        
        # Prepare labels
        y = self._prepare_labels(y)
        
        # Split data
        n_train = int(len(X) * (1 - validation_split))
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create base XGBoost classifier
        base_clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # Multi-output wrapper
        self.model = MultiOutputClassifier(base_clf)
        
        # Train
        print("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if validation_split > 0 else None
        )
        
        # Evaluate
        train_pred = self.model.predict_proba(X_train)
        val_pred = self.model.predict_proba(X_val)
        
        # Convert predict_proba output to proper format
        if isinstance(train_pred, list):
            train_pred = np.array([p[:, 1] for p in train_pred]).T
            val_pred = np.array([p[:, 1] for p in val_pred]).T
        
        metrics = {
            'train_auc': roc_auc_score(y_train, train_pred, average='macro'),
            'val_auc': roc_auc_score(y_val, val_pred, average='macro'),
            'train_ap': average_precision_score(y_train, train_pred, average='macro'),
            'val_ap': average_precision_score(y_val, val_pred, average='macro')
        }
        
        print(f"Training complete. Val AUC: {metrics['val_auc']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict resistance mechanisms.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted probabilities (n_samples × n_mechanisms)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        pred_proba = self.model.predict_proba(X)
        
        # Convert to proper format
        if isinstance(pred_proba, list):
            pred_proba = np.array([p[:, 1] for p in pred_proba]).T
        
        return pred_proba
    
    def predict_classes(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict resistance mechanism classes.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        threshold : float
            Probability threshold
            
        Returns
        -------
        np.ndarray
            Binary predictions
        """
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns
        -------
        pd.DataFrame
            Feature importance per mechanism
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = []
        for i, clf in enumerate(self.model.estimators_):
            feat_imp = clf.feature_importances_
            for j, (feat_name, imp) in enumerate(zip(self.feature_names, feat_imp)):
                importances.append({
                    'mechanism': self.resistance_mechanisms[i],
                    'feature': feat_name,
                    'importance': imp
                })
        
        return pd.DataFrame(importances)
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'resistance_mechanisms': self.resistance_mechanisms,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.resistance_mechanisms = data['resistance_mechanisms']
        params = data.get('params', {})
        self.n_estimators = params.get('n_estimators', 100)
        self.max_depth = params.get('max_depth', 6)
        self.learning_rate = params.get('learning_rate', 0.1)

