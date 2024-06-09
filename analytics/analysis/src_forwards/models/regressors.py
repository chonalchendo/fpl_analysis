from dataclasses import dataclass

from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor


@dataclass
class Regressors:
    rf: RandomForestRegressor = RandomForestRegressor()
    gbr: GradientBoostingRegressor = GradientBoostingRegressor()
    hgbr: HistGradientBoostingRegressor = HistGradientBoostingRegressor()
    xgb: XGBRegressor = XGBRegressor()

    @property
    def models(self) -> list[RegressorMixin]:
        return [self.rf, self.gbr, self.hgbr, self.xgb]


regressors = Regressors()
