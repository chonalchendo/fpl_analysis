import httpx
import pandas as pd
import streamlit as st
from core.settings import API_URL


def main() -> None:
    st.title("Foobtall player market value predictor")

    url = f"{API_URL}/dropdowns/get"

    # get data for dropdowns
    response = httpx.get(url).json()
    countries = response.get("countries", [])
    leagues = response.get("leagues", [])
    positions = response.get("positions", [])

    # create query form for user input
    with st.form(key="query_form"):
        country = st.selectbox("Country", countries, placeholder="Select a country")
        league = st.selectbox("League", leagues, placeholder="Select a league")
        position = st.selectbox("Position", positions, placeholder="Select a position")
        submit_button = st.form_submit_button(label="Submit")

    # query the API and display the results
    if submit_button:
        with st.spinner("Querying..."):
            query = {}
            if country is not None:
                query["country"] = country
            if league is not None:
                query["league"] = league
            if position is not None:
                query["position"] = position

            if not query:
                st.warning("Please select at least one filter")

            response = httpx.get(
                f"{API_URL}/value_prediction/predict",
                params=query,
            ).json()

        data = pd.DataFrame(response)
        st.dataframe(data)


if __name__ == "__main__":
    main()
