"""
Time series cross-validation utilities using grouped-timeserie-cv.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    from grouptimeseriescv import GroupTimeSeriesSplit
    GROUPTIMESERIESCV_AVAILABLE = True
except ImportError:
    GROUPTIMESERIESCV_AVAILABLE = False
    warnings.warn(
        "grouped-timeserie-cv not available. Using sklearn TimeSeriesSplit as fallback.",
        ImportWarning
    )
    
from sklearn.model_selection import TimeSeriesSplit


class GroupedTimeSeriesValidator:
    """
    Cross-validation for time series data with proper temporal splits.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 group_column: Optional[str] = None):
        """
        Initialize the validator.
        
        Parameters:
        -----------
        n_splits : int
            Number of splits for cross-validation
        test_size : int, optional
            Size of test set (in time units)
        gap : int
            Gap between train and test sets
        group_column : str, optional
            Column name for grouping (e.g., target ID)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.group_column = group_column
        
        if GROUPTIMESERIESCV_AVAILABLE and group_column:
            self.cv_splitter = GroupTimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap
            )
        else:
            # Fallback to regular TimeSeriesSplit
            self.cv_splitter = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap
            )
    
    def validate_model(self, 
                      model,
                      X: np.ndarray,
                      y: np.ndarray,
                      groups: Optional[np.ndarray] = None,
                      scoring_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Perform time series cross-validation on a model.
        
        Parameters:
        -----------
        model : sklearn model
            Model to validate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        groups : np.ndarray, optional
            Group labels for GroupTimeSeriesSplit
        scoring_metrics : List[str], optional
            List of scoring metrics to compute
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        results = {metric: [] for metric in scoring_metrics}
        results['fold_results'] = []
        
        # Determine split method based on available libraries and parameters
        if GROUPTIMESERIESCV_AVAILABLE and groups is not None:
            splits = self.cv_splitter.split(X, y, groups)
        else:
            splits = self.cv_splitter.split(X, y)
        
        fold = 0
        for train_idx, test_idx in splits:
            fold += 1
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            fold_results = {
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_positive_rate': y_train.mean(),
                'test_positive_rate': y_test.mean()
            }
            
            # Compute scoring metrics
            if 'accuracy' in scoring_metrics:
                acc = accuracy_score(y_test, y_pred)
                results['accuracy'].append(acc)
                fold_results['accuracy'] = acc
            
            if 'precision' in scoring_metrics:
                prec = precision_score(y_test, y_pred, zero_division=0)
                results['precision'].append(prec)
                fold_results['precision'] = prec
            
            if 'recall' in scoring_metrics:
                rec = recall_score(y_test, y_pred, zero_division=0)
                results['recall'].append(rec)
                fold_results['recall'] = rec
            
            if 'f1' in scoring_metrics:
                f1 = f1_score(y_test, y_pred, zero_division=0)
                results['f1'].append(f1)
                fold_results['f1'] = f1
            
            if 'roc_auc' in scoring_metrics and hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                    results['roc_auc'].append(auc)
                    fold_results['roc_auc'] = auc
                except ValueError:
                    # Handle case where only one class is present in test set
                    results['roc_auc'].append(np.nan)
                    fold_results['roc_auc'] = np.nan
            
            results['fold_results'].append(fold_results)
            
            print(f"Fold {fold}: Accuracy={fold_results.get('accuracy', 'N/A'):.3f}, "
                  f"F1={fold_results.get('f1', 'N/A'):.3f}")
        
        # Calculate summary statistics
        summary = {}
        for metric in scoring_metrics:
            if metric in results and results[metric]:
                scores = np.array(results[metric])
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    summary[f'{metric}_mean'] = valid_scores.mean()
                    summary[f'{metric}_std'] = valid_scores.std()
                    summary[f'{metric}_scores'] = valid_scores.tolist()
                else:
                    summary[f'{metric}_mean'] = np.nan
                    summary[f'{metric}_std'] = np.nan
                    summary[f'{metric}_scores'] = []
        
        results['summary'] = summary
        
        return results
    
    def temporal_validation_split(self, 
                                 df: pd.DataFrame,
                                 time_column: str = 'time',
                                 test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a temporal train-test split.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        time_column : str
            Name of the time column
        test_ratio : float
            Proportion of data to use for testing
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training and testing DataFrames
        """
        # Sort by time
        df_sorted = df.sort_values(time_column)
        
        # Calculate split point
        split_point = int(len(df_sorted) * (1 - test_ratio))
        
        train_df = df_sorted.iloc[:split_point].copy()
        test_df = df_sorted.iloc[split_point:].copy()
        
        print(f"Temporal split - Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        print(f"Train time range: {train_df[time_column].min():.2f} - {train_df[time_column].max():.2f}")
        print(f"Test time range: {test_df[time_column].min():.2f} - {test_df[time_column].max():.2f}")
        
        return train_df, test_df
    
    def grouped_validation_split(self,
                                df: pd.DataFrame,
                                group_column: str,
                                test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a grouped train-test split (e.g., by target ID).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        group_column : str
            Column to group by
        test_ratio : float
            Proportion of groups to use for testing
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training and testing DataFrames
        """
        # Get unique groups
        unique_groups = df[group_column].unique()
        n_test_groups = max(1, int(len(unique_groups) * test_ratio))
        
        # Randomly select test groups
        np.random.seed(42)  # For reproducibility
        test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)
        
        # Split data
        test_mask = df[group_column].isin(test_groups)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"Grouped split - Train: {len(train_df)} samples from {len(unique_groups) - n_test_groups} groups")
        print(f"Test: {len(test_df)} samples from {n_test_groups} groups")
        
        return train_df, test_df
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of predictions.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_prob : np.ndarray, optional
            Predicted probabilities
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC if probabilities are available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['roc_auc'] = np.nan
        
        # Class distribution
        metrics['positive_rate_true'] = y_true.mean()
        metrics['positive_rate_pred'] = y_pred.mean()
        
        # Sample counts
        metrics['n_samples'] = len(y_true)
        metrics['n_positive_true'] = int(y_true.sum())
        metrics['n_positive_pred'] = int(y_pred.sum())
        
        return metrics
    
    def print_evaluation_report(self, metrics: Dict[str, float]):
        """
        Print a formatted evaluation report.
        
        Parameters:
        -----------
        metrics : Dict[str, float]
            Dictionary of evaluation metrics
        """
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        
        print(f"Dataset Size: {metrics.get('n_samples', 'N/A')}")
        print(f"True Positives: {metrics.get('n_positive_true', 'N/A')}")
        print(f"Predicted Positives: {metrics.get('n_positive_pred', 'N/A')}")
        print(f"True Positive Rate: {metrics.get('positive_rate_true', 0):.3f}")
        print(f"Predicted Positive Rate: {metrics.get('positive_rate_pred', 0):.3f}")
        
        print("\nClassification Metrics:")
        print(f"Accuracy:  {metrics.get('accuracy', 0):.3f}")
        print(f"Precision: {metrics.get('precision', 0):.3f}")
        print(f"Recall:    {metrics.get('recall', 0):.3f}")
        print(f"F1-Score:  {metrics.get('f1', 0):.3f}")
        
        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            print(f"ROC AUC:   {metrics.get('roc_auc', 0):.3f}")
        
        print("="*50)