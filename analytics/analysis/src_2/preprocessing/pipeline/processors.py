from sklearn.pipeline import make_pipeline

from analysis.src_2.preprocessing.feature_engineering.feature_creation import (
    create_league_signed_from_col,
    create_penalty_taker_col,
    create_years_since_signed_col,
    create_youth_player_col,
)
from analysis.src_2.preprocessing.transformers.functional import ApplyFunction
# from analysis.src_2.preprocessing.transformers.drop_features import DropFeatures
from analysis.src_2.preprocessing.transformers.imputers import (
    BoolImputer,
    CustomImputer,
    GroupbyImputer,
)


youth_player_feat = ApplyFunction(
    feature="is_youth_player", apply_func=create_youth_player_col, return_type="pandas"
)


league_signed_from_feat = ApplyFunction(
    feature="league_signed_from",
    apply_func=create_league_signed_from_col,
)

signed_from_impute = CustomImputer(
    features="signed_from", value="Unknown", return_type="pandas"
)

signed_year_impute = CustomImputer(
    features="signed_year", impute_by_col="season", return_type="pandas"
)


pen_taker_feat = ApplyFunction(
    feature="is_penalty_taker",
    apply_func=create_penalty_taker_col,
    feature_names_out="feat",
)
year_since_signed_feat = ApplyFunction(
    feature="years_since_signed",
    apply_func=create_years_since_signed_col,
    feature_names_out="feat",
)

signing_fee_impute = make_pipeline(
    BoolImputer(
        feature="signing_fee_euro_mill",
        value=0,
        condition_col="is_youth_player",
        condition=True,
        return_type="pandas",
    ),
    GroupbyImputer(
        feature="signing_fee_euro_mill",
        groupby=["league", "age_range"],
        strategy="median",
        return_type="pandas",
    ),
)

stats_imputer = CustomImputer(
    groupby=["league", "age_range", "position"],
    strategy="median",
    return_type="pandas",
)

# cols_to_drop = [
#     "player_id",
#     "rk",
#     "general_pos",
#     "pos",
#     "nation",
#     "comp",
#     "born",
#     "annual_wages_euros",
#     "player",
#     "age_range",
# ]

# drop_features = DropFeatures(features=cols_to_drop)
