from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)

ridge = ("ridge", Ridge(random_state=42))
rf = ("rf", RandomForestRegressor(random_state=42))
gbr = ("gbr", GradientBoostingRegressor(random_state=42))
xgb = ("xgb", XGBRegressor(random_state=42))
abr = ("abr", AdaBoostRegressor(random_state=42))
kr = ("kr", KernelRidge())
br = ("br", BayesianRidge())
hgb = ("hgb", HistGradientBoostingRegressor(random_state=42))
stacked = (
    "stacked",
    StackingRegressor(
        [ridge, rf, gbr, xgb, abr, kr, br, hgb],
        final_estimator=XGBRegressor(random_state=42),
        n_jobs=-1,
    ),
)


models = [
    ridge,
    rf,
    gbr,
    xgb,
    abr,
    kr,
    br,
    hgb,
    stacked,
]
