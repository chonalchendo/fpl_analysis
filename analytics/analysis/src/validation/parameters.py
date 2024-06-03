from __future__ import annotations
import numpy as np
from scipy import stats


def params_dict() -> list[tuple[str, dict]]:
    params = [
        # ("ridge", ridge_param),
        ("xgb", xgb_params),
        ("rf", rf_params),
        ("gbr", gbr_params),
        # ("abr", abr_param),
        # ("kr", kr_params),
        # ("br", br_param),
        ("hgb", hgb_params),
    ]
    return sorted(params, key=lambda x: x[0])


ridge_param = {
    'alpha': stats.loguniform(1e-3, 1e3),
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'],
}

rf_params = {
    'n_estimators': stats.randint(50, 1000),  # Number of trees in the forest
    'max_depth': stats.randint(2, 100),  # Maximum depth of the tree
    'min_samples_split': stats.randint(2, 20),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': stats.randint(1, 20),  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

gbr_params = {
    'n_estimators': stats.randint(100, 500),  # Number of boosting stages
    'learning_rate': stats.uniform(0.01, 0.3),  # Learning rate shrinks the contribution of each tree
    'max_depth': [3, 4, 5, 6, 7],  # Maximum depth of the individual regression estimators
    'min_samples_split': stats.randint(2, 20),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': stats.randint(1, 20),  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'],  # Number of features to consider when looking for the best split
    'subsample': stats.uniform(0.5, 0.5),  # Subsample ratio of the training instance
}

xgb_params = {
    'n_estimators': stats.randint(50, 200),  # Number of boosting rounds
    'max_depth': stats.randint(3, 10),  # Maximum depth of a tree
    'learning_rate': stats.uniform(0.01, 0.3),  # Learning rate
    'subsample': stats.uniform(0.5, 0.5),  # Subsample ratio of the training instance
    'colsample_bytree': stats.uniform(0.5, 0.5),  # Subsample ratio of columns when constructing each tree
    'gamma': stats.uniform(0, 1),  # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'reg_alpha': stats.uniform(0, 1),  # L1 regularization term on weights
    'reg_lambda': stats.uniform(0, 1)  # L2 regularization term on weights
}

abr_param = {
    'n_estimators': stats.randint(50, 200),
    'learning_rate': stats.uniform(0.1, 1.0),
}

kr_params = {
    'alpha': stats.uniform(1, 10),  # Regularization parameter
    'kernel': ['linear', 'polynomial'],  # Kernel type
    'gamma': stats.uniform(0.1, 1),  # Kernel coefficient for 'rbf' and 'polynomial'
    'coef0': stats.uniform(0, 5)  # Independent term in kernel function
}

br_param = {
    'alpha_1': np.linspace(1e-6, 1e-4, 100),
    'alpha_2': np.linspace(1e-6, 1e-4, 100),
    'lambda_1': np.linspace(1e-6, 1e-4, 100),
    'lambda_2': np.linspace(1e-6, 1e-4, 100),
}

hgb_params = {
    'learning_rate': stats.uniform(0.01, 0.3),  # Learning rate
    'max_iter': stats.randint(50, 200),  # Maximum number of iterations
    'max_depth': [3, 4, 5, 6, None],  # Maximum depth of the individual regression estimators
    'min_samples_leaf': stats.randint(1, 20),  # Minimum number of samples required to be at a leaf node
    'l2_regularization': stats.uniform(0, 1)  # L2 regularization term on weights
}



