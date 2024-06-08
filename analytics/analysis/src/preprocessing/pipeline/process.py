from sklearn.pipeline import make_pipeline

from analysis.src.preprocessing.transformers.imputers import (
    BoolImputer,
    CustomImputer,
    GroupbyImputer,
)

signed_from_impute = CustomImputer(
    features="signed_from", value="Unknown", return_type="pandas"
)

signed_year_impute = CustomImputer(
    features="signed_year", impute_by_col="season", return_type="pandas"
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
