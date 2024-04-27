from sklearn.base import RegressorMixin
from sklearn.model_selection import RandomizedSearchCV

from analysis.gcp.storage import gcp

# class ModelValidator:
#     def __init__(
#         self,
#         model: RegressorMixin,
#         param_distributions: dict,
#         n_iter: int = 100,
#         cv: int = 5,
#         n_jobs: int = -1,
#     ):
#         self.model = model
#         self.param_distributions = param_distributions
#         self.n_iter = n_iter
#         self.cv = cv
#         self.n_jobs = n_jobs

#     def tune(self, X, y):
#         tuner = RandomizedSearchCV(
#             self.model,
#             self.param_distributions,
#             n_iter=self.n_iter,
#             cv=self.cv,
#             n_jobs=self.n_jobs,
#         )
#         tuner.fit(X, y)
#         return tuner.best_params_


def validate() -> None:
    # load in validation data
    valid = gcp.read_df_from_bucket(
        bucket_name="values_validation_data", file_name="valid_set_2023.csv"
    )
    # load in model
    # split data into X and y
    # run through validation pipeline
    # return tuned models and results
    # save results to gcp

    pass


if __name__ == "__main__":
    validate()
