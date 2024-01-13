from feature_pipeline.etl.fpl.src import extract, transform, load
from feature_pipeline.utilities import utils
from feature_pipeline.utilities.storage import gcp

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

    logger.info("Saving data to FPL data to GCP bucket")
    gcp.write_blob_to_bucket("fpl_db", "player_stats.csv", data)

    logger.info("ETL pipeline complete")


# if __name__ == "__main__":
#     run_etl()
