import pandas as pd


class ColumnSelector:
    def __init__(
        self,
        columns: list[str] | None = None,
        dtype_include: str | None = None,
        dtype_exclude: str | None = None,
    ) -> None:
        self.columns = columns
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, X: pd.DataFrame) -> list[str]:
        if not hasattr(X, "iloc"):
            raise ValueError("Input must be a Pandas DataFrame")

        df_row = X.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(
                include=self.dtype_include, exclude=self.dtype_exclude
            )

        cols = df_row.columns.tolist()

        if self.columns is not None:
            cols = cols + self.columns

        return cols
