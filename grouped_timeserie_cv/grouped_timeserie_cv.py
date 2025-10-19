from .data_preprocessor import DataPreprocessor
from .hyperparameter_tuning import HyperparameterTuning
from .learning_curve_calculator import LearningCurveCalculator
from .model_evaluator import ModelEvaluator
from .plotter import Plotter
from .data_class.cross_validation_result import CrossValidationResult
from sklearn.metrics import confusion_matrix

class GroupedTimeSerieCV:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.grid_search_executor = HyperparameterTuning()
        self.learning_curve_calculator = LearningCurveCalculator()
        self.model_evaluator = ModelEvaluator()
        self.plotter = Plotter()

    def classify(self, data, pipelines, param_grids, frequency ='D', datetime_column ='DateTime', label_column = 'Label',  scoring='accuracy'):
        print("Beginning classification with grouped time series cross-validation...")
        X, y, groups = self.preprocessor.preprocess_data(data, frequency, datetime_column, label_column)
        best_model, best_params, selected_feature_names = self.grid_search_executor.perform_grid_search(X, y, groups, pipelines, param_grids, scoring)
        train_sizes, train_mean, train_std, test_mean, test_std = self.learning_curve_calculator.compute_learning_curve(best_model, X, y, groups, scoring)
        y_pred, incorrect_dates = self.model_evaluator.evaluate_model(best_model, X, y, groups, data)
        confusion_matrices = confusion_matrix(y, y_pred, labels=sorted(y.unique()))

        print("Best model:", best_model.named_steps['model'].__class__.__name__)
        print("Best parameters for best results:", best_params)
        print("Selected features:", selected_feature_names)

        return CrossValidationResult(
            confusion_matrices=confusion_matrices,
            class_labels=sorted(y.unique()),
            train_sizes=train_sizes,
            train_mean=train_mean,
            train_std=train_std,
            test_mean=test_mean,
            test_std=test_std,
            best_model=best_model,
            selected_feature_names=selected_feature_names,
            best_params=best_params,
            incorrect_dates=incorrect_dates,
            actual_values=y,
            predicted_values=y_pred
        )

    def predict(self, data, pipelines, param_grids, frequency ='D', datetime_column ='DateTime', label_column = 'Label', scoring='neg_mean_squared_error'):
        X, y, groups = self.preprocessor.preprocess_data(data, frequency, datetime_column, label_column)
        best_model, best_params, selected_feature_names = self.grid_search_executor.perform_grid_search(X, y, groups, pipelines, param_grids, scoring)
        train_sizes, train_mean, train_std, test_mean, test_std = self.learning_curve_calculator.compute_learning_curve(best_model, X, y, groups, scoring)
        y_pred, incorrect_dates = self.model_evaluator.evaluate_model(best_model, X, y, groups, data)

        print("Best model:", best_model.named_steps['model'].__class__.__name__)
        print("Best parameters for best results:", best_params)
        print("Selected features:", selected_feature_names)
        
        return CrossValidationResult(
            confusion_matrices=None,
            class_labels=None,
            train_sizes=train_sizes,
            train_mean=train_mean,
            train_std=train_std,
            test_mean=test_mean,
            test_std=test_std,
            best_model=best_model,
            selected_feature_names=selected_feature_names,
            best_params=best_params,
            incorrect_dates=incorrect_dates,
            actual_values=y,
            predicted_values=y_pred
        )