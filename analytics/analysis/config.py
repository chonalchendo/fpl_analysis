from scipy.stats import randint

# -------------------------------- Grid Search ------------------------------- #

# define the parameter grid to search for the best combination of hyperparameters
grid_params = [
    # try 12 (3×4) combinations of hyperparameters
    {
        "n_estimators": [3, 10, 30, 50, 100, 200],
        "max_features": [2, 4, 6, 8, 10, 12, 14, 16],
    },
    # then try 6 (2×3) combinations with bootstrap set as False
    {
        "bootstrap": [False],
        "n_estimators": [3, 10],
        "max_features": [2, 3, 4],
    },
]


# -------------------------------- Randomized Search ------------------------------- #

random_params = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=16),
}
