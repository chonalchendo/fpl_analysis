import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Literal
from explore.src.data.stats_dfs import StatsData


@dataclass
class StatsVisuals:
    """Class to create visualisations for the stats data."""

    _data: pd.DataFrame

    def avgs_time_series(self, groupby: str, columns: list[str]) -> None:
        """Create time series plots for average stats.

        Args:
            groupby (str): column to groupby e.g. 'comp'.
            columns (list[str]): columns to visualise
        """
        data = self._data.copy()

        avgs_df = StatsData(data).avgs_df(groupby=groupby, columns=columns)
        avgs_cols = [col for col in avgs_df.columns if "avg_" in col]

        fig, axes = plt.subplots(figsize=(20, 30), nrows=8, ncols=3)

        # remove empty subplots
        for i in range(len(avgs_cols), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for stat, ax in zip(avgs_cols, axes.flatten()):
            sns.lineplot(data=avgs_df, x="season", y=stat, hue=groupby, ax=ax)
            ax.set_title(stat)
            ax.get_legend().remove()
            plt.setp(ax.get_xticklabels(), rotation=15)

        # set legend
        handles, labels = axes.flatten()[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(1, 1),
            title="Competition",
        )

        plt.tight_layout()
        plt.show()

    def top_per_season(
        self,
        columns: list[str],
        league: Literal[
            "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "all"
        ] = "all",
        figsize: tuple[int, int] = (20, 30),
        nrows: int = 8,
        ncols: int = 3,
    ) -> None:
        """Create bar plots for top stats per season.

        Args:
            columns (list[str]): columns to visualise.
            figsize (tuple[int, int], optional): figure size. Defaults to (20, 30).
            nrows (int, optional): number of rows. Defaults to 8.
            ncols (int, optional): number of columns. Defaults to 3.
        """
        data = self._data.copy().reset_index(drop=True)

        if league != "all":
            data = data[data["comp"] == league].reset_index(drop=True)

        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

        # remove empty subplots
        for i in range(len(columns), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for stat, ax in zip(columns, axes.flatten()):
            indexes = data.groupby("season")[stat].idxmax().values
            df = data.iloc[indexes]
            df["player_season"] = df["player"] + " " + df["season"]

            plots = sns.barplot(
                data=df,
                y="player_season",
                x=stat,
                # palette="Blues_r",
                # hue="player",
                legend=False,
                ax=ax,
            )
            for bar in plots.patches:
                plots.annotate(
                    format(bar.get_width(), ".2f"),
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    ha="left",
                    va="center",
                    size=10,
                    xytext=(5, 0),
                    textcoords="offset points",
                )
            ax.set_title(f"Top {stat} for each season")
            ax.set_ylabel("Player & Season")

        plt.tight_layout()
        plt.show()
