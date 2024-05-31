from processing.gcp.files import gcs
from processing.src.preprocessing import _fbref, _transfermarkt
from processing.utilities.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    fbref_bucket = "fbref_db"
    transfermarkt_bucket = "transfermarkt_db"

    fbref_wages = gcs.list_bucket(fbref_bucket, include="wages")
    fbref_stats = gcs.list_bucket(fbref_bucket, exclude="wages")
    transfermarkt_player = gcs.list_bucket(transfermarkt_bucket, include="player")
    transfermarkt_team = gcs.list_bucket(transfermarkt_bucket, include="team")

    save = "no"

    logger.info("Running preprocessing for fbref wages")

    for file in fbref_wages:
        logger.info(file)
        _fbref.run_wages(blob=file, output_blob=f"processed_{file}", save=save)

    logger.info("Running preprocessing for fbref stats")

    for file in fbref_stats:
        logger.info(file)
        _fbref.run_stats(blob=file, output_blob=f"processed_{file}", save=save)

    logger.info("Preprocessing for transfermarkt player data")

    for file in transfermarkt_player:
        logger.info(f"Processing: {file}")
        _transfermarkt.run_players(
            blob=file, output_blob=f"processed_{file}", save=save
        )

    logger.info("Preprocessing for transfermarkt team data")

    for file in transfermarkt_team:
        logger.info(f"Processing: {file}")
        _transfermarkt.run_teams(blob=file, output_blob=f"processed_{file}", save=save)


if __name__ == "__main__":
    main()
