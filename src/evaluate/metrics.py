"""
Evaluation Metrics Module

Provides comprehensive metrics for resistance prediction evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix, classification_report,
    cohen_kappa_score
)
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_classification_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None,
                                    average: str = 'macro') -> Dict:
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    average : str
        Averaging strategy for multi-label
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['support'] = support
    
    # AUC and AP if probabilities provided
    if y_pred_proba is not None:
        try:
            if y_true.ndim == 1:
                # Binary or multiclass
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, average=average)
                metrics['ap'] = average_precision_score(y_true, y_pred_proba, average=average)
            else:
                # Multi-label
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, average=average)
                metrics['ap'] = average_precision_score(y_true, y_pred_proba, average=average)
        except Exception as e:
            print(f"Warning: Could not calculate AUC/AP: {e}")
            metrics['auc'] = 0.0
            metrics['ap'] = 0.0
    
    # Cohen's kappa
    try:
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    except:
        metrics['kappa'] = 0.0
    
    return metrics


def calculate_time_to_event_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   event_times: np.ndarray,
                                   censored: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate time-to-event (survival) metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True event indicators
    y_pred : np.ndarray
        Predicted risk scores
    event_times : np.ndarray
        Event times
    censored : np.ndarray, optional
        Censoring indicators
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    try:
        from sksurv.metrics import concordance_index_censored
        
        if censored is None:
            censored = ~y_true.astype(bool)
        
        c_index, _, _, _, _ = concordance_index_censored(
            censored.astype(bool),
            y_true.astype(bool),
            y_pred
        )
        
        return {'c_index': c_index}
    except ImportError:
        print("scikit-survival not available. Install with: pip install scikit-survival")
        return {'c_index': 0.0}
    except Exception as e:
        print(f"Error calculating C-index: {e}")
        return {'c_index': 0.0}


def calculate_subgroup_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              subgroups: pd.DataFrame,
                              y_pred_proba: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Calculate metrics for demographic/clinical subgroups.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    subgroups : pd.DataFrame
        Subgroup assignments
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
        
    Returns
    -------
    pd.DataFrame
        Metrics per subgroup
    """
    results = []
    
    for col in subgroups.columns:
        for subgroup_value in subgroups[col].unique():
            mask = subgroups[col] == subgroup_value
            
            if mask.sum() == 0:
                continue
            
            subgroup_y_true = y_true[mask]
            subgroup_y_pred = y_pred[mask]
            subgroup_y_pred_proba = y_pred_proba[mask] if y_pred_proba is not None else None
            
            metrics = calculate_classification_metrics(
                subgroup_y_true,
                subgroup_y_pred,
                subgroup_y_pred_proba
            )
            
            results.append({
                'subgroup_column': col,
                'subgroup_value': subgroup_value,
                'n_samples': mask.sum(),
                **metrics
            })
    
    return pd.DataFrame(results)


def generate_evaluation_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None,
                              subgroups: Optional[pd.DataFrame] = None) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    class_names : list, optional
        Class names
    subgroups : pd.DataFrame, optional
        Subgroup assignments
        
    Returns
    -------
    dict
        Comprehensive evaluation report
    """
    report = {}
    
    # Overall metrics
    report['overall'] = calculate_classification_metrics(
        y_true, y_pred, y_pred_proba
    )
    
    # Per-class metrics if multi-class
    if y_true.ndim == 1 and len(np.unique(y_true)) > 2:
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        
        report['per_class'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    
    # Subgroup analysis
    if subgroups is not None:
        report['subgroups'] = calculate_subgroup_metrics(
            y_true, y_pred, subgroups, y_pred_proba
        )
    
    return report

