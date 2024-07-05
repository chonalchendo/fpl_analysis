import streamlit as st


def configure_app() -> None:
    st.set_page_config(
        page_title="Football Player Market Value Predictor",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def configure_overview() -> None:
    st.title("Football Player Market Value Predictor")
    st.markdown(
        "This app predicts the market value of football players based on their country, league, and position."
    )
