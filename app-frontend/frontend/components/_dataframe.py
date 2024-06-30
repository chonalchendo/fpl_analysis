import pandas as pd
import streamlit as st


def create_dataframe(data: pd.DataFrame) -> st.dataframe:
    return st.dataframe(data)
