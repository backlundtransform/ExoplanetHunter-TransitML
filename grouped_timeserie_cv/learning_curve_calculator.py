from sklearn.model_selection import GroupKFold, learning_curve
import numpy as np

class LearningCurveCalculator:
    @staticmethod
    def compute_learning_curve(best_model, X, y, groups, scoring):
        group_kfold = GroupKFold(n_splits=len(np.unique(groups)))
      
        train_sizes, train_scores, test_scores = learning_curve(best_model, X, y,
                                                                cv=group_kfold,
                                                                train_sizes=np.linspace(0.1, 1.0, 10),
                                                                groups=groups,
                                                                scoring=scoring)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        return train_sizes, train_mean, train_std, test_mean, test_std