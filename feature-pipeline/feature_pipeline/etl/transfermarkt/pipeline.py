from icecream import ic
from feature_pipeline.etl.transfermarkt.src import extract, transform, load
from feature_pipeline.apis.transfermarkt import competition_ids, competition_names
from feature_pipeline.utilities.utils import get_logger
from feature_pipeline.utilities.storage import gcp


logger = get_logger(__name__)


def run_leagues_etl() -> None:
    """Extracts, transforms and loads league data from Transfermarkt"""

    logger.info("Beginning transfermarkt leagues etl pipeline")

    for comp_id, comp_name in zip(competition_ids, competition_names):
        logger.info(f"Extracting league data for competition: {comp_id}")
        df = extract.leagues_data(comp_id=comp_id, comp_name=comp_name)
        ic(df)
        logger.info(f"Extraction for {comp_id} complete")

        logger.info(f"Transforming league data for competition: {comp_id}")
        df = transform.league_data(df)
        ic(df)

        logger.info(f"Saving league data for competition: {comp_id}")
        # load to google cloud sql or s3 bucket
        load.to_sql_database(
            data=df, table_name=f"{comp_name}_league_data", database="transfermarkt"
        )

        logger.info(f"Saving league data for {comp_id} to GCP bucket")
        gcp.write_blob_to_bucket(
            bucket_name="transfermarkt_db",
            blob_name=f"{comp_name}_team_data.csv",
            data=df,
        )

    logger.info("Finished Transfermarkt leagues etl pipeline")


def run_player_vals_etl() -> None:
    """Extracts, transforms and loads player valuations data from Transfermarkt"""

    logger.info("Beginning transfermarkt player valuations etl pipeline")

    for league in competition_names:
        logger.info(f"Extracting player valuations data for league: {league}")
        df = extract.player_valuations(league=league)

        # check all teams are present for each season
        check = df.groupby("season")["team"].nunique()
        logger.info(f"Number of teams per season: {check}")

        # transform data
        logger.info("Transforming player valuations data")
        df = transform.market_data(df)
        ic(df)

        logger.info("Saving player valuations data")
        # load to postgres database
        load.to_sql_database(
            data=df, table_name=f"{league}_player_valuations", database="transfermarkt"
        )

        logger.info(f"Saving player valuations data for {league} to GCP bucket")
        gcp.write_blob_to_bucket(
            bucket_name="transfermarkt_db",
            blob_name=f"{league}_player_valuations.csv",
            data=df,
        )

    logger.info("Finished Trnasfermarkt player valuations etl pipeline")


# if __name__ == "__main__":
#     run_leagues_etl()
#     run_player_vals_etl()
