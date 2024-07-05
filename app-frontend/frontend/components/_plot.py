import pandas as pd
import plotly.express as px
import streamlit as st


def create_plot(data: pd.DataFrame):
    if data is not None:
        data.columns = data.columns.str.replace("_", " ").str.title()

        data2 = data.melt(
            id_vars="Player",
            value_vars=["Market Value", "Prediction"],
            value_name="value",
        )

        fig = px.bar(data2, y="Player", x="value", color="variable", barmode="group")

        fig.update_traces(
            texttemplate="%{value:.2f}",
            textposition="outside",
            marker_line_color="rgb(8,48,107)",
            marker_line_width=1.5,
            opacity=0.6,
        )
        fig.update_layout(
            title="Player Market Value vs Prediction",
            xaxis_title="Value",
            yaxis_title="Player",
            legend_title="Value Type",
            bargap=0.1,
        )
        return st.plotly_chart(fig, use_container_width=True)
