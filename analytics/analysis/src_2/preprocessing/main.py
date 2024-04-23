import numpy as np
from rich import print
from sklearn.base import RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from analysis.gcp.storage import gcp
from analysis.src_2.preprocessing.pipeline.build import (
    cleaning_pipeline,
    feature_transformer,
)


def regression_pipeline(
    model: RegressorMixin, preprocessor: Pipeline, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("transformer", transformer),
            ("model", model),
        ]
    )


target_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(
        func=np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one"
    ),
    StandardScaler(),
)

cols_to_drop = [
    "player_id",
    "rk",
    "general_pos",
    "pos",
    "nation",
    "comp",
    "born",
    "annual_wages_euros",
    "player",
    "age_range",
]


def main() -> None:
    df = gcp.read_df_from_bucket(
        bucket_name="wage_vals_stats", blob_name="standard.csv"
    )
    # create train, valid, test splits
    train_set = df.loc[df["season"] != 2023]
    valid_test_set = df.loc[df["season"] == 2023]

    valid_set, test_set = train_test_split(
        valid_test_set,
        test_size=0.5,
        stratify=valid_test_set["position"],
        random_state=42,
    )

    X_train, y_train = (
        train_set.drop("market_value_euro_mill", axis=1),
        train_set["market_value_euro_mill"],
    )
    

    num_features = [
        col
        for col in X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if col not in cols_to_drop
    ]
    cat_features = [
        col
        for col in X_train.select_dtypes(include=["object"]).columns.tolist()
        if col not in cols_to_drop
    ]

    clean_pipe = cleaning_pipeline(drop_cols=cols_to_drop)
    transformer = feature_transformer(
        num_features=num_features, cat_features=cat_features
    )

    preprocessor = Pipeline(
        [("preprocessor", clean_pipe), ("transformer", transformer)], verbose=True
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("target", target_pipe),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X_train, y_train):
        clone_model = clone(pipeline["model"])
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]
        
        print(X_train_folds.isnull().sum().sum())
        

        print("cleaning training data")
        # clean train data
        y_train_folds = pipeline["target"].fit_transform(
            y_train_folds.to_numpy().reshape(-1, 1)
        )
        X_train_folds = pipeline["preprocess"].fit_transform(
            X_train_folds, y_train_folds.ravel()
        )
        
        print(np.isnan(X_train_folds).sum())
        

        print("cleaning test data")
        # clean test data
        y_test_fold = pipeline["target"].fit_transform(
            y_test_fold.to_numpy().reshape(-1, 1)
        )
        X_test_fold = pipeline["preprocess"].fit_transform(
            X_test_fold, y_test_fold.ravel()
        )
        
        print(X_test_fold.shape)
        print(X_train_folds.shape)

        print("fitting model")
        clone_model.fit(X_train_folds, y_train_folds.ravel())

        y_pred = clone_model.predict(X_test_fold)

        y_pred = np.expm1(y_pred)
        y_test_fold = np.expm1(y_test_fold)

        print(mean_absolute_error(y_test_fold, y_pred))
        print(mean_squared_error(y_test_fold, y_pred, squared=False))
        print(r2_score(y_test_fold, y_pred))


if __name__ == "__main__":
    main()
