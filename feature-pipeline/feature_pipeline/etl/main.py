from feature_pipeline.etl.fbref.pipeline import run_stats_etl, run_wages_etl
from feature_pipeline.etl.fpl.pipeline import run_etl as run_fpl_etl
from feature_pipeline.etl.transfermarkt.pipeline import (
    run_leagues_etl,
    run_player_vals_etl,
)
from feature_pipeline.utilities.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main script that handles running all etl pipelines"""

    logger.info("Starting all ETL pipelines")

    logger.info("Running FPL ETL pipeline")
    run_fpl_etl()

    logger.info("Running Transfermarkt ETL pipelines")
    run_leagues_etl()
    run_player_vals_etl()

    logger.info("Running FBref ETL pipelines")
    run_stats_etl()
    run_wages_etl()

    logger.info("All ETL pipelines complete")


if __name__ == "__main__":
    main()
