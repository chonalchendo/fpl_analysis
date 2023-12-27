from google.cloud import storage
from analysis.settings import SETTINGS
from dataclasses import dataclass
from typing import Any
import pandas as pd
import joblib
from analysis.utils import get_logger

logger = get_logger(__name__)


@dataclass
class GCP:
    """Dataclass for interacting with Google Cloud Platform."""

    bucket_name: str = SETTINGS["GOOGLE_CLOUD_BUCKET_NAME"]
    bucket_project: str = SETTINGS["GOOGLE_CLOUD_PROJECT"]
    json_creds_path: str = SETTINGS["GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"]

    def get_gcp_bucket(self) -> storage.Bucket:
        """Get the GCP bucket object.

        Returns:
            storage.Bucket: GCP bucket object
        """
        try:
            logger.info("Getting GCP bucket")
            storage_client = storage.Client.from_service_account_json(
                json_credentials_path=self.json_creds_path, project=self.bucket_project
            )
            bucket = storage_client.get_bucket(self.bucket_name)
            logger.info("GCP bucket retrieved")
            return bucket
        except Exception as e:
            logger.error(f"Error getting GCP bucket: {e}")

    def write_blob_to_bucket(self, blob_name: str, model: dict[str, Any]) -> None:
        """Writes a file to the bucket.

        Args:
            blob_name (str): name of blob in bucket
            model (dict[str, Any]): dictionary of model information
        """
        bucket = self.get_gcp_bucket()
        
        logger.info(f"Creating blob: {blob_name}")
        blob = bucket.blob(blob_name)
        
        with blob.open("wb") as f:
            joblib.dump(model, f)
        logger.info(f"{model['model']} has been saved to {blob_name}")

    def read_blob_from_bucket(self, blob_name: str) -> pd.DataFrame | None:
        """Reads a file from the bucket.

        Args:
            blob_name (str): name of blob in bucket

        Returns:
            pd.DataFrame | None: dataframe of blob contents or None if blob does not exist
        """
        bucket = self.get_gcp_bucket()
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return None

        with blob.open("rb") as f:
            return pd.read_pickle(f)


gcp = GCP()
