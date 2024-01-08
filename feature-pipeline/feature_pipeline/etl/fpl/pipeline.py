from feature_pipeline.etl.fpl.src import extract, transform, load
from feature_pipeline.utilities import utils

logger = utils.get_logger(__name__)


def run_etl() -> None:
    """Main Script to run the ETL pipeline."""
    logger.info("Extracting data from pickle file")
    data, metadata_ = extract.from_file()

    logger.info("Transforming data")
    data = transform.transform(data)
    logger.info("Data transformed")

    logger.info("Loading data into PostgreSQL database")
    load.to_sql_database(data, "player_stats")
    logger.info("Data loaded into PostgreSQL database")

    logger.info(metadata_)
    logger.info("ETL pipeline complete")


if __name__ == "__main__":
    run_etl()
