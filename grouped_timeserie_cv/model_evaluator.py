from sklearn.model_selection import GroupKFold, cross_val_predict
import numpy as np
class ModelEvaluator:
    @staticmethod
    def evaluate_model(best_model, X, y, groups, data):
        group_kfold = GroupKFold(n_splits=len(np.unique(groups)))
        y_pred = cross_val_predict(best_model, X, y, cv=group_kfold, groups=groups)
        incorrect_indices = [i for i, (true, pred) in enumerate(zip(y, y_pred)) if true != pred]
        incorrect_dates = data.iloc[incorrect_indices]['DateTime'].dt.date.unique()
        return y_pred, incorrect_dates