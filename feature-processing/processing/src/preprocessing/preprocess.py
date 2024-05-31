from processing.gcp.files import gcs
from processing.src.preprocessing import _fbref, _transfermarkt
from processing.utilities.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    fbref_bucket = "fbref_db"
    transfermarkt_bucket = "transfermarkt_db"

    fbref_files = gcs.list_bucket(fbref_bucket)
    transfermarkt_player_files = gcs.list_bucket(transfermarkt_bucket, filter="player")
    transfermarkt_team_files = gcs.list_bucket(transfermarkt_bucket, filter="team")

    save = "no"

    logger.info("Running preprocessing for fbref ")

    for file in fbref_files:
        logger.info(file)
        _fbref.run(blob=file, output_blob=f"processed_{file}", save=save)

    logger.info("Preprocessing for transfermarkt player data")

    for file in transfermarkt_player_files:
        logger.info(f"Processing: {file}")
        _transfermarkt.clean_player_df(
            blob=file, output_blob=f"processed_{file}", save=save
        )

    logger.info("Preprocessing for transfermarkt team data")

    for file in transfermarkt_team_files:
        logger.info(f"Processing: {file}")
        _transfermarkt.clean_team_df(
            blob=file, output_blob=f"processed_{file}", save=save
        )


if __name__ == "__main__":
    main()
