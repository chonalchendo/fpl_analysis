import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from explore.utilities.utils import get_logger

logger = get_logger(__name__)


@dataclass
class StatisticPlots:
    _data: pd.DataFrame

    def _export_plot(self, filename: str) -> None:
        """Exports plot to file.

        Args:
            filename (str): filename to save plot to
        """
        date = datetime.date.today().strftime("%Y-%m-%d")
        file_dir = Path.cwd() / "reports" / "figures" / date
        path = file_dir / filename + ".png"

        if not file_dir.exists():
            file_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(path, bbox_inches="tight")
        logger.info(f"Plot saved to {path}")

    def relationships(
        self,
        y: str,
        X: list[str],
        hue: str = None,
        palette: str = None,
        nrows: int = 4,
        ncols: int = 4,
        figsize: tuple[int, int] = (15, 15),
        save_to: str = None,
    ) -> None:
        """Plots relationships between y and X.

        Args:
            y (str): dependent variable
            X (list[str]): list of independent variables
            hue (str, optional): column to hue by. Defaults to None.
            palette (str, optional): colour palette. Defaults to None.
            nrows (int, optional): plot grid rows. Defaults to 4.
            ncols (int, optional): plot grid columns. Defaults to 4.
            save_to (str, optional): filename to save to. Defaults to None.
        """
        # make sure y is not in X
        X = [col for col in X if col != y]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        # remove empty subplots
        for i in range(len(X), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

        for stat, ax in zip(X, axes.flatten()):
            sns.scatterplot(
                data=self._data,
                x=stat,
                y=y,
                hue=hue,
                palette=palette,
                ax=ax,
                legend=True,
            )
            ax.set_title(f"{y} vs {stat}")
            ax.set_xlabel(stat)
            ax.set_ylabel(y)

            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        plt.suptitle(f"{y} vs Related Stats", fontsize=20)
        plt.tight_layout()
        plt.show()

        if save_to:
            self._export_plot(save_to)

    def correlation_matrix(
        self,
        vars: list[str],
        figsize: tuple[int, int] = (16, 6),
        save_to: str | None = None,
    ) -> None:
        """Plots a correlation matrix.

        Args:
            save_to (str | None): filename to save plot to
        """
        df = self._data[vars]

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))
        heatmap = sns.heatmap(
            df.corr(numeric_only=True),
            mask=mask,
            vmin=-1,
            vmax=1,
            annot=True,
            cmap="bwr",
        )
        heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 18}, pad=16)
        plt.show()

        if save_to:
            self._export_plot(save_to)

    def dependent_correlations(
        self,
        y_var: str,
        vars: list[str],
        figsize: tuple[int, int] = (4, 6),
        save_to: str | None = None,
    ) -> None:
        """Plot correlation matrix for dependent variable.

        Args:
            y_var (str): dependent variable
            save_to (str | None): filename to save plot to
        """
        df = self._data[vars]

        plt.figure(figsize=figsize)
        heatmap = sns.heatmap(
            df.corr()[[y_var]].sort_values(by=y_var, ascending=False),
            vmin=-1,
            vmax=1,
            annot=True,
            cmap="bwr",
        )
        heatmap.set_title(
            f"Features Correlating with {y_var.capitalize()}",
            fontdict={"fontsize": 18},
            pad=16,
        )
        plt.show()

        if save_to:
            self._export_plot(save_to)

    def distribution(self, x: str, save_to: str | None) -> None:
        """Plots a distribution plot of x.

        Args:
            x (str): variable to plot
            save_to (str | None): filename to save plot to
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(data=self._data, x=x, kde=True)
        plt.title(f"{x.capitalize()} Distribution", fontsize=20)
        plt.xlabel(x)
        plt.ylabel("Count")
        plt.show()

        if save_to:
            self._export_plot(save_to)

    def all_distributions(self, save_to: str | None = None) -> None:
        """Plots a distribution plot of all variables.

        Args:
            save_to (str | None): filename to save plot to
        """
        data = self._data.select_dtypes(include=np.number)
        data.hist(figsize=(16, 20), bins=20, xlabelsize=8, ylabelsize=8)
        plt.tight_layout()
        plt.show()

        if save_to:
            self._export_plot(save_to)
