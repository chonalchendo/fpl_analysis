from rich import print

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

    print(fbref_files)
    print(transfermarkt_player_files)
    print(transfermarkt_team_files)

    # logger.info("Running preprocessing for fbref ")
    # _fbref.run(blob="fbref.csv", output_blob="processed_.csv")
    #
    # logger.info("Running preprocessing for transfermarkt ")
    # _transfermarkt.run()
    #


if __name__ == "__main__":
    main()
