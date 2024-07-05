import pandas as pd
import streamlit as st


def colour_diff(val):
    colour = "green" if val > 0 else "red"
    return f"color: {colour}"


def create_dataframe(data: pd.DataFrame) -> st.dataframe:
    if data is not None:
        data.columns = data.columns.str.replace("_", " ").str.title()

        data["Value Difference"] = data["Prediction"] - data["Market Value"]

        return st.dataframe(
            data.style.map(colour_diff, subset=["Value Difference"]),
            hide_index=True,
        )
