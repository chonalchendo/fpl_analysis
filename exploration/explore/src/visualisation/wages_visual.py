from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from explore.src.data.wages_dfs import WagesData


@dataclass
class WagesVisuals:
    """Class to create visualisations for the wages data."""

    _data: pd.DataFrame

    def top_earners(self) -> None:
        """Top 10 weekly earners per season in the dataset."""
        seasons = self._data["season"].unique().tolist()

        fig, axes = plt.subplots(5, 2, figsize=(12, 12))

        for i in range(len(seasons), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for stat, ax in zip(seasons, axes.flatten()):
            data = (
                self._data.loc[self._data["season"] == stat]
                .sort_values("weekly_wages_euros", ascending=False)
                .head(10)
            )
            sns.barplot(
                data=data, y="player", x="weekly_wages_euros", ax=ax, palette="Blues_r"
            )
            ax.set_title(f"{stat} Weekly Wages (Euros)")
            ax.set_xlabel("Weekly Wages (Euros)")
            # change x axis spacing
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)

            for i, v in enumerate(data["weekly_wages_euros"]):
                ax.text(
                    v + 0.5,
                    i + 0.10,
                    f"€{v:,.0f}",
                    ha="left",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.show()

    def total_wage_avgs(
        self,
        column: str,
        figsize: tuple[int, int] = (20, 12),
        wages: Literal[
            "weekly_wages_euros", "annual_wages_euros"
        ] = "weekly_wages_euros",
    ) -> None:
        """Total wage averages by column.

        Args:
            column (str): column to group by.
            figsize (tuple[int, int], optional): Figure size. Defaults to (20, 12).
            wages (Literal[ &quot;weekly_wages_euros&quot;, &quot;annual_wages_euros&quot; ], optional): Select a wage variable. Defaults to "weekly_wages_euros".
        """
        data = WagesData(self._data, groupby=column).run_total_data()

        fig, axes = plt.subplots(figsize=figsize)

        fig = sns.barplot(
            data=data,
            y=column,
            x=wages,
            color="royalblue",
            ax=axes,
            palette="Blues_r",
        )

        for i, v in enumerate(data[wages]):
            axes.text(
                v + 0.5,
                i + 0.10,
                f"€{v:,.0f}",
                ha="left",
                va="center",
                color="black",
                fontsize=12,
            )

        plt.xlabel(f"{wages.strip('_euros').capitalize()} (Euros)")
        plt.ylabel(f"{column.capitalize()}")
        plt.title(
            f"{wages.strip('_euros').capitalize()} (Euros) by {column.capitalize()}"
        )

    def top_earners_season_var(
        self, yaxis_var: str, hue: str, figsize: tuple[int, int] = (22, 22)
    ) -> None:
        """Top earners at each club by season.

        Args:
            yaxis_var (str): Select the y axis variable.
            hue (str): Select the hue variable.
            figsize (tuple[int, int], optional): Define the figure size. Defaults to (22, 22).
        """
        # top earner at each club
        seasons = self._data["season"].unique().tolist()

        # palette
        colors = sns.color_palette("tab10")
        palette = dict(zip(self._data[hue].unique(), colors))

        fig, axes = plt.subplots(5, 2, figsize=figsize)

        for i in range(len(seasons), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for season, ax in zip(seasons, axes.flatten()):
            data = self._data.loc[(self._data["season"] == season)]

            data_2 = data.loc[
                data.groupby("squad")["weekly_wages_euros"].idxmax()
            ].sort_values("weekly_wages_euros", ascending=False)

            sns.barplot(
                data=data_2,
                y=yaxis_var,
                x="weekly_wages_euros",
                hue=hue,
                ax=ax,
                palette=palette,
            )
            ax.legend(
                loc="lower right",
                fancybox=True,
                shadow=True,
            )

            ax.set_title(f"{season} Top Earner at Each Club")

            for i, v in enumerate(data_2["weekly_wages_euros"]):
                ax.text(
                    v + 0.5,
                    i + 0.10,
                    f"€{v:,.0f}",
                    ha="left",
                    va="center",
                    color="black",
                    fontsize=10,
                )
        plt.tight_layout()

    def time_series(self, groupby: str, figsize: tuple[int, int] = (20, 12)) -> None:
        """Time series plots for weekly and annual wages.

        Args:
            groupby (str): Group by variable.
            figsize (tuple[int, int], optional): Define the figure size. Defaults to (20, 12).
        """
        data = WagesData(self._data, groupby=groupby).run_seasonal_data()

        cols = data.filter(like="weekly").columns

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        for col, ax in zip(cols, axes.flatten()):
            sns.lineplot(
                data=data,
                x="season",
                y=col,
                hue=groupby,
                ax=ax,
                style=groupby,
                markers=True,
                palette="tab10",
            )

            ax.set_title(f"{col.capitalize()} by {groupby.capitalize()}")
            ax.set_xlabel("Season")
            ax.set_ylabel(f"{col.capitalize()} (Euros)")
            # ax.legend(loc="right", bbox_to_anchor=(1, 1), ncol=2)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=2)

        plt.tight_layout()
        plt.show()

    def pct_change_plots(
        self, pct_change_col: str, figsize: tuple[int, int] = (20, 20)
    ) -> None:
        """Percentage change plots for weekly and annual wages by club over time.

        Args:
            pct_change_col (str): Select the percentage change column.
            figsize (tuple[int, int], optional): Define the figure size. Defaults to (20, 20).
        """
        data = WagesData(self._data.copy(), groupby="squad").run_seasonal_data()

        teams = data["squad"].unique().tolist()

        fig, axes = plt.subplots(nrows=8, ncols=4, figsize=figsize)

        for i in range(len(teams), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for team, ax in zip(teams, axes.flatten()):
            df = data.loc[data["squad"] == team]
            colors = ["royalblue" if x > 0 else "red" for x in df[pct_change_col]]

            sns.barplot(data=df, x="season", y=pct_change_col, ax=ax, palette=colors)

            ax.set_title(f"{team}")
            ax.set_xlabel("Season")
            ax.set_ylabel("% Change")
            ax.set_xticklabels(df["season"], rotation=45)

        plt.suptitle(f"% Change in {pct_change_col.capitalize()}")
        plt.tight_layout()
