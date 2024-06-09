from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def forwards_pipeline(model: RegressorMixin) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include="object"),
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        [
            ("preprocessor", transformer),
            ("model", model),
        ]
    )
