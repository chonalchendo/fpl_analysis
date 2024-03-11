import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from explore.utilities.utils import get_logger
from explore.src.data.valuations_dfs import PlayerValData, PlayerValues

logger = get_logger(__name__)


@dataclass
class PlayerValsPlots:
    """Class to plot visualisations for player values data."""

    _data: pd.DataFrame

    def valuation_plot(self, column: str) -> go.Figure:
        """Plot the average signing fee and market value by column

        Args:
            column (str): column to group by

        Returns:
            go.Figure: plotly figure
        """
        data = self._data.copy()
        df = PlayerValues(_data=data, column=column).pipeline()

        if column == "country":
            top_countries = data["country"].value_counts().index[:10]
            df = df[df["country"].isin(top_countries)]

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(
            go.Bar(
                x=df[column],
                y=df["signing_fee_euro_mill"],
                name="Signing Fee",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df[column],
                y=df["market_value_euro_mill"],
                name="Market Value",
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            title=f"Average Signing Fee and Market Value by {column.capitalize()}",
            xaxis_title=f"{column.capitalize()}",
            yaxis_title="Average Value",
            legend_title="Value Type",
            barmode="group",
            xaxis={"categoryorder": "total descending"},
        )
        return fig

    def var_totals_plot(self, column: str) -> go.Figure:
        """Plot the total players by chosen variable

        Args:
            column (str): column to group by

        Returns:
            go.Figure: plotly figure
        """
        data = self._data.copy()
        df = PlayerValData().season_count_df(data, column)

        # plot time series of total players by chosen variable
        line_df = df.sort_values("2023", ascending=False)[1:11]
        line_dff = pd.melt(
            line_df,
            id_vars="country",
            value_vars=line_df.columns[1:],
        )

        fig = px.line(
            line_dff,
            x="variable",
            y="value",
            color="country",
            title="Top 10 Countries by Total Players",
        )
        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Total Players",
            legend_title=f"{column.capitalize()}",
        )
        return fig

    def plot_diff_val_paid(self, column: str) -> go.Figure:
        """Plot the difference in value by a chosen variable.

        Args:
            column (str): column to group by

        Returns:
            go.Figure: plotly figure
        """
        df = PlayerValData().diff_values_df(self._data, column)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Difference in Value", "Difference in Value (%)"),
        )

        fig.add_trace(
            go.Bar(
                x=df[column],
                y=df["diff_value_paid"],
                name="Difference in Value",
                marker_color=df["color"],
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df[column],
                y=df["diff_value_paid_perc"],
                name="Difference in Value (%)",
                marker_color=df["color"],
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="Difference in value by team",
            xaxis_title=f"{column.capitalize()}",
            yaxis_title="Average Value",
            legend_title="Value Type",
            barmode="group",
            xaxis={"categoryorder": "total descending"},
            showlegend=False,
        )
        return fig

    def plot_season_apps(self) -> go.Figure:
        """Plot the number of season appearances by team.

        Returns:
            go.Figure: plotly figure
        """
        df = PlayerValData().count_season_apps_df(self._data)

        # plot the appearances
        fig = px.bar(
            data_frame=df,
            x="teams",
            y="appearances",
            title="Number of Season Appearances by Team",
        )

        fig.update_layout(
            xaxis_title="Team",
            yaxis_title="Appearances",
        )
        return fig

    def plot_value_signings(self) -> None:
        """Plot the value for money signings by season."""
        df = PlayerValData().value_signings_df(self._data)
        fig, axes = plt.subplots(5, 2, figsize=(10, 10))

        # remove empty subplots
        for i in range(len(df["season"].unique().tolist()), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for stat, ax in zip(df["season"].unique().tolist(), axes.flatten()):
            sns.barplot(
                data=df.loc[df["season"] == stat]
                .sort_values(by="diff_sign_fee_mv", ascending=False)
                .head(10),
                y="player",
                x="diff_sign_fee_mv",
                orient="h",
                ax=ax,
            )
            ax.set_title(f"Season {stat}")
            ax.set_xlabel("Difference between signing fee and market value")

        plt.suptitle("Value for money signings by season")
        plt.tight_layout()
        plt.show()


@dataclass
class TeamValsPlots:
    _data: pd.DataFrame

    def top_10_plot(self, column: str, ascend: bool = False) -> plt.Axes:
        """Plot the top 10 values by column.

        Args:
            column (str): column to sort by
            ascend (bool, optional): sort method. Defaults to False.

        Returns:
            plt.Axes: matplotlib axes
        """
        df = self._data.copy()
        data = df.sort_values(by=column, ascending=ascend).head(10)

        # plot the values
        fig, ax = plt.subplots(figsize=(8, 6))
        fig = sns.barplot(
            data=data, y="team_season", x=column, ax=ax, color="royalblue"
        )

        if "value" in column:
            ax.bar_label(
                ax.containers[-1], fmt="€%.2fm", label_type="center", color="white"
            )
        elif "pct" in column:
            ax.bar_label(
                ax.containers[-1], fmt="%.2f%%", label_type="center", color="white"
            )
        elif "age" in column:
            ax.bar_label(
                ax.containers[-1], fmt="%.2f", label_type="center", color="white"
            )
        else:
            ax.bar_label(
                ax.containers[-1], fmt="%d", label_type="center", color="white"
            )
        return fig

    def time_series_plot(self, column: str, seasons: int = 1) -> plt.Axes:
        """Plot a time series of a column for teams that have appeared in a
        certain number of seasons.

        Args:
            df (pd.DataFrame): dataframe to plot
            column (str): column to plot
            seasons (int, optional): season to filter. Defaults to 1.

        Returns:
            plt.Axes: matplotlib axes
        """

        # count season apps for each team
        season_apps = PlayerValData().count_season_apps_df(self._data)

        # teams that have appeared in every season
        teams = season_apps[season_apps["appearances"] >= seasons]["teams"].values

        # filter the data
        data = self._data[self._data["team"].isin(teams)]

        # plot the data
        fig, ax = plt.subplots(figsize=(12, 6))

        fig = sns.lineplot(
            data,
            x="season",
            y=column,
            hue="team",
            style="team",
            ax=ax,
            palette="tab10",
            markers=True,
        )

        title = column.split("_")[0].capitalize()

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.xlabel("Season")
        plt.ylabel(f"{title} Squad Value (€m)")
        plt.title(
            f"{title} squad value over time for teams that have appeared in {seasons} seasons or more"
        )
        return fig
