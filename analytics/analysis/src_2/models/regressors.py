from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)


class Models:
    def __init__(self) -> None:
        self.ridge = ("ridge", Ridge(random_state=42))
        self.rf = ("rf", RandomForestRegressor(random_state=42))
        self.gbr = ("gbr", GradientBoostingRegressor(random_state=42))
        self.xgb = ("xgb", XGBRegressor(random_state=42))
        self.abr = ("abr", AdaBoostRegressor(random_state=42))
        self.kr = ("kr", KernelRidge())
        self.br = ("br", BayesianRidge())
        self.hgb = ("hgb", HistGradientBoostingRegressor(random_state=42))

    @property
    def get_models(self) -> list[tuple[str, RegressorMixin]]:
        models = [
            # self.ridge,
            self.xgb,
            self.rf,
            self.gbr,
            self.hgb,
            # self.abr,
            # self.kr,
            # self.br,
        ]
        return sorted(models, key=lambda x: x[0])

    @property
    def get_stacked(self) -> tuple[str, RegressorMixin]:
        return (
            "stacked",
            StackingRegressor(
                [
                    self.ridge,
                    self.rf,
                    self.gbr,
                    self.xgb,
                    self.abr,
                    self.kr,
                    self.br,
                    self.hgb,
                ],
                final_estimator=self.xgb[1],
                n_jobs=-1,
            ),
        )
