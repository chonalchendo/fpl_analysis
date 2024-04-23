from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline

from analysis.src_2.preprocessing.transformers.drop_features import DropFeatures
from analysis.src_2.preprocessing.feature_engineering.dimension_reduction import (
    ClusterSimilarity,
)
from analysis.src_2.preprocessing.pipeline.processors import (
    signed_from_impute,
    signed_year_impute,
    signing_fee_impute,
    stats_imputer,
    youth_player_feat,
    pen_taker_feat,
    year_since_signed_feat,
)


num_transformer = make_pipeline(StandardScaler())

cat_transformer = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="most_frequent"),
    OneHotEncoder(sparse_output=False),
)

target_pipeline = make_pipeline(
    TargetEncoder(),
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

dimension_reduction = FeatureUnion(
    [
        ("cluster", ClusterSimilarity()),
        ("pca", PCA(n_components=0.95)),
    ]
)


def feature_transformer(
    num_features: list[str], cat_features: list[str]
) -> ColumnTransformer:

    target_cols = ["squad", "country", "signed_from"]
    cat_features = [col for col in cat_features if col not in target_cols]

    return ColumnTransformer(
        [
            ("is_pen_taker", pen_taker_feat, ["penalty_kicks_attempted"]),
            ("year_since_signed", year_since_signed_feat, ["signed_year", "season"]),
            ("target", target_pipeline, target_cols),
            ("decomp", dimension_reduction, ["age", "weekly_wages_euros"]),
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
    )


def cleaning_pipeline(drop_cols: list[str]) -> Pipeline:
    return make_pipeline(
        youth_player_feat,
        signed_from_impute,
        signed_year_impute,
        signing_fee_impute,
        stats_imputer,
        DropFeatures(features=drop_cols),
    )

