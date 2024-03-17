from typing import Literal

from processing.src.preprocess.fbref import Clean as FbrefClean
from processing.src.preprocess.transfermarkt import CleanTeam, CleanPlayer
from processing.src.preprocess.utils import generate_unique_id
from processing.utilities.utils import get_logger
from processing.gcp.storage import gcp

logger = get_logger(__name__)


def process_data(
    data_source: Literal["fbref", "transfermarkt"],
) -> None:
    """Processes the data from the specified source.

    Args:
        data_source (Literal[fbref, transfermarkt]): Select the data source to process.
    """
    # list all blobs in the specified bucket
    blobs = gcp.list_blobs(f"{data_source}_db")

    for blob in blobs:

        try:
            logger.info(f"Processing {blob}")
            df = gcp.read_df_from_bucket(
                bucket_name=f"{data_source}_db", blob_name=blob
            )

            logger.info(f"Cleaning {blob}")

            # specify cleaning pipeline
            if data_source == "fbref":
                cleaned_data = FbrefClean(df).pipeline()
            elif "team" in blob:
                cleaned_data = CleanTeam(df).pipeline()
            else:
                cleaned_data = CleanPlayer(df).pipeline()

            # generate a unique id for each player
            if "team" not in blob:
                player_id_map = generate_unique_id()
                cleaned_data["player_id"] = cleaned_data["player"].map(player_id_map)

            # initialise new bucket and blob
            new_bucket = f"processed_{data_source}_db"
            new_blob = f"processed_{blob}"

            logger.info(f"Writing {new_blob} to {new_bucket}")
            gcp.write_df_to_bucket(
                bucket_name=new_bucket,
                blob_name=new_blob,
                data=cleaned_data,
            )
        except Exception as e:
            logger.error(f"Error processing {blob}: {e}")
            continue

    logger.info(f"Finished processing {data_source} data")


def main() -> None:
    """Main function to process the data."""
    logger.info("Processing fbref data")
    process_data("fbref")

    logger.info("Processing transfermarkt data")
    process_data("transfermarkt")

    logger.info("Finished processing data")


if __name__ == "__main__":
    main()
