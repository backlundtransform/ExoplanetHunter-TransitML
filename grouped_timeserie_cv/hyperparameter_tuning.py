from sklearn.model_selection import GroupKFold, GridSearchCV
import numpy as np

class HyperparameterTuning:
    @staticmethod
    def perform_grid_search(X, y, groups, pipelines, param_grids, scoring):
        group_kfold = GroupKFold(n_splits=len(np.unique(groups)))
        best_model = None
        best_score = float('-inf')
        best_params = None
        selected_feature_names = None

        for pipeline, param_grid in zip(pipelines, param_grids):
            try:
                grid_search = GridSearchCV(pipeline, param_grid, cv=group_kfold, n_jobs=-1, scoring=scoring)
                grid_search.fit(X, y, groups=groups)
                print("Process model:", grid_search.best_estimator_.named_steps['model'].__class__.__name__)
                print("Score:", grid_search.best_score_)

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    selected_features = best_model.named_steps['selector'].get_support()
                    all_features = X.columns
                    selected_feature_names = all_features[selected_features].tolist()
            except Exception as e:
                print(f"An error occurred during grid search: {e}")

        return best_model, best_params, selected_feature_names