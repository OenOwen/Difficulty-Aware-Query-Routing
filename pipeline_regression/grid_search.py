from sklearn.model_selection import GridSearchCV


def perform_grid_search(pipeline, X_train, y_train, param_grid):
    """
    Perform grid search to find the best hyperparameters for the given pipeline.

    Args:
        pipeline: The machine learning pipeline to optimize.
        X_train: Training features.
        y_train: Training labels.
        param_grid: A dictionary specifying the hyperparameters and their values to search.
    """

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
