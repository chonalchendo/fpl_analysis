from icecream import ic
from feature_pipeline.etl.fbref.src import extract, transform, load
from feature_pipeline.apis.fbref import tables, stats, comp_ids, comp_names
from feature_pipeline.utilities.utils import get_logger
from feature_pipeline.utilities.storage import gcp

logger = get_logger(__name__)


def run_stats_etl() -> None:
    """Run etl pipeline for fbref player stats"""

    logger.info("Beginning fbref player stats etl pipeline")

    for stat, table in zip(stats, tables):
        logger.info(f"Extracting data for stat: {stat} and table: {table}")
        df = extract.player_stats(stat=stat, table=table)

        logger.info(f"Transforming data for stat: {stat} and table: {table}")
        df = transform.player_stats(df=df, table=table)

        logger.info(f"Saving data for stat: {stat} and table: {table}")
        load.to_sql_database(data=df, table_name=table, database="fbref")

        logger.info(
            f"Data loaded to local Postgres database for stat: {stat} and table: {table}"
        )

        logger.info(f"Saving data for stat: {stat} and table: {table} to GCP bucket")
        gcp.write_blob_to_bucket(
            bucket_name="fbref_db", blob_name=f"{table}.csv", data=df
        )

    logger.info(
        "All fbref stats data loaded to local Postgres database and stored in GCP bucket"
    )


def run_wages_etl() -> None:
    """Run etl pipeline for fbref player wages"""

    logger.info("Beginning fbref player wages etl pipeline")

    for comp_id, comp_name in zip(comp_ids, comp_names):
        logger.info(f"Extracting wage data for competition: {comp_name}")
        df = extract.player_wages(comp_id=comp_id, comp_name=comp_name)

        logger.info(f"Transforming wage data for competition: {comp_name}")
        df = transform.wages(df=df, comp=comp_name)
        ic(df)

        logger.info(f"Saving wage data for competition: {comp_name}")
        load.to_sql_database(data=df, table_name=f"{comp_name}-wages", database="fbref")

        logger.info(f"Wage data loaded to local Postgres database for {comp_name}")

        logger.info(f"Saving wage data for competition: {comp_name} to GCP bucket")
        gcp.write_blob_to_bucket(
            bucket_name="fbref_db", blob_name=f"{comp_name}-wages.csv", data=df
        )

    logger.info(
        "All fbref wage data loaded to local Postgres database and stored in GCP bucket"
    )


# if __name__ == "__main__":
#     # run both etl pipelines
#     run_stats_etl()
#     run_wages_etl()
