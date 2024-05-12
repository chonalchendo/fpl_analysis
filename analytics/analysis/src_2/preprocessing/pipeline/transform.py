from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from analysis.src_2.preprocessing.feature_engineering.dimension_reduction import (
    ClusterSimilarity,
)
from analysis.src_2.preprocessing.transformers.functional import ApplyFunction
from analysis.src_2.preprocessing.feature_engineering.feature_creation import (
    create_league_signed_from_col,
    create_penalty_taker_col,
    create_years_since_signed_col,
    create_youth_player_col,
)


youth_player_feat = ApplyFunction(
    feature="is_youth_player", apply_func=create_youth_player_col, return_type="pandas"
)


league_signed_from_feat = ApplyFunction(
    feature="league_signed_from",
    apply_func=create_league_signed_from_col,
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

num_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

cat_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(sparse_output=False),
)

target_encoder = make_pipeline(
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
