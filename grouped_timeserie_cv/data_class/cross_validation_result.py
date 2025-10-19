from dataclasses import dataclass
import numpy as np


@dataclass
class CrossValidationResult:
    confusion_matrices: np.ndarray
    class_labels: list
    train_sizes: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    test_mean: np.ndarray
    test_std: np.ndarray
    best_model: object
    selected_feature_names: list
    best_params: dict
    incorrect_dates: np.ndarray
    actual_values: np.ndarray
    predicted_values: np.ndarray