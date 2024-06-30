import streamlit as st
from models import Query
from services import get_dropdowns


def configure_sidebar() -> Query | None:
    dropdowns = get_dropdowns()
    # create query form for user input

    with st.sidebar.form(key="query_form"):
        st.write("## Filters")

        country = st.selectbox(
            "Country", dropdowns.countries, placeholder="Select a country", index=None
        )
        league = st.selectbox(
            "League", dropdowns.leagues, placeholder="Select a league", index=None
        )
        position = st.selectbox(
            "Position", dropdowns.positions, placeholder="Select a position", index=None
        )
        submit_button = st.form_submit_button(label="Submit")

    # query the API and display the results
    if submit_button:
        return Query(country=country, league=league, position=position)
