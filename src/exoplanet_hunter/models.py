"""
Machine learning models for exoplanet transit detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path


class TransitClassifier:
    """
    A collection of machine learning models for exoplanet transit detection.
    """
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        """
        Initialize the transit classifier.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'svm', 'logistic', 'neural_network')
        random_state : int
            Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on the specified type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            )
        elif self.model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                C=1.0,
                random_state=self.random_state,
                max_iter=1000
            )
        elif self.model_type == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                random_state=self.random_state,
                max_iter=500
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, df: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare features for training or prediction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        feature_columns : List[str], optional
            Specific columns to use as features
            
        Returns:
        --------
        np.ndarray
            Feature matrix
        """
        if feature_columns is None:
            # Use default feature columns (exclude time and target columns)
            exclude_cols = ['time', 'target', 'has_transit', 'original_flux']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature columns for consistency
        if self.feature_columns is None:
            self.feature_columns = feature_columns
        
        # Ensure all required columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X = df[self.feature_columns].values
        
        # Handle any remaining NaN values
        if np.any(np.isnan(X)):
            print("Warning: NaN values found in features, filling with column means")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        return X
    
    def create_transit_labels(self, time: np.ndarray, flux: np.ndarray,
                             transit_threshold: float = 0.01,
                             window_size: int = 5) -> np.ndarray:
        """
        Create transit labels based on flux dips.
        
        Parameters:
        -----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Normalized flux array
        transit_threshold : float
            Threshold for defining a transit (flux dip)
        window_size : int
            Window size for local minima detection
            
        Returns:
        --------
        np.ndarray
            Binary labels (1 for transit, 0 for no transit)
        """
        labels = np.zeros(len(flux))
        
        # Find local minima that are below the threshold
        for i in range(window_size, len(flux) - window_size):
            # Check if this is a local minimum
            is_local_min = all(flux[i] <= flux[i-j] for j in range(1, window_size+1)) and \
                          all(flux[i] <= flux[i+j] for j in range(1, window_size+1))
            
            # Check if the dip is significant enough
            if is_local_min and flux[i] < (1.0 - transit_threshold):
                labels[i] = 1
        
        return labels
    
    def train(self, df: pd.DataFrame, target_column: str = 'has_transit',
              feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        feature_columns : List[str], optional
            Specific columns to use as features
            
        Returns:
        --------
        Dict[str, Any]
            Training results and metrics
        """
        print(f"Training {self.model_type} model...")
        
        # Prepare features and target
        X = self.prepare_features(df, feature_columns)
        y = df[target_column].values
        
        print(f"Training data shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X)
        train_probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        results = {
            'training_accuracy': (train_predictions == y).mean(),
            'classification_report': classification_report(y, train_predictions),
            'confusion_matrix': confusion_matrix(y, train_predictions)
        }
        
        if train_probabilities is not None:
            results['training_auc'] = roc_auc_score(y, train_probabilities)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        print(f"Training completed. Accuracy: {results['training_accuracy']:.3f}")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = np.zeros_like(predictions)
        
        return predictions, probabilities
    
    def cross_validate(self, df: pd.DataFrame, target_column: str = 'has_transit',
                      cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        target_column : str
            Name of the target column
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        Dict[str, Any]
            Cross-validation results
        """
        X = self.prepare_features(df)
        y = df[target_column].values
        
        scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring=scoring)
        
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_folds': cv_folds,
            'scoring': scoring
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.random_state = model_data['random_state']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


class EnsembleTransitClassifier:
    """
    Ensemble classifier combining multiple models for robust transit detection.
    """
    
    def __init__(self, model_types: List[str] = None, random_state: int = 42):
        """
        Initialize the ensemble classifier.
        
        Parameters:
        -----------
        model_types : List[str], optional
            List of model types to include in ensemble
        random_state : int
            Random state for reproducibility
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic']
        
        self.model_types = model_types
        self.random_state = random_state
        self.models = {}
        self.is_trained = False
        
        # Initialize individual models
        for model_type in model_types:
            self.models[model_type] = TransitClassifier(
                model_type=model_type, 
                random_state=random_state
            )
    
    def train(self, df: pd.DataFrame, target_column: str = 'has_transit') -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
            
        Returns:
        --------
        Dict[str, Any]
            Training results for all models
        """
        print("Training ensemble models...")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model_results = model.train(df, target_column)
            results[model_name] = model_results
        
        self.is_trained = True
        return results
    
    def predict(self, df: pd.DataFrame, method: str = 'voting') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data
        method : str
            Ensemble method ('voting' or 'averaging')
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Ensemble predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from all models
        for model in self.models.values():
            pred, prob = model.predict(df)
            all_predictions.append(pred)
            all_probabilities.append(prob)
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        if method == 'voting':
            # Majority voting
            ensemble_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, 
                arr=all_predictions
            )
        else:  # averaging
            # Average probabilities and threshold at 0.5
            ensemble_predictions = (np.mean(all_probabilities, axis=0) > 0.5).astype(int)
        
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
        
        return ensemble_predictions, ensemble_probabilities