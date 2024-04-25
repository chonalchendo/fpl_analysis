from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

models = [
    ("lr", LinearRegression()),
    ("lasso", Lasso(random_state=42)),
    ("ridge", Ridge(random_state=42)),
    ("elasticnet", ElasticNet(random_state=42)),
    ("rf", RandomForestRegressor(random_state=42)),
    ("gbr", GradientBoostingRegressor(random_state=42)),
    ("xgb", XGBRegressor(random_state=42)),
]