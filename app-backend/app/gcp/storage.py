from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from google.cloud import storage

from app.core.settings import SETTINGS
from app.utilities.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GCP:
    """Dataclass for interacting with Google Cloud Platform."""

    # bucket_name: str = SETTINGS["GOOGLE_CLOUD_BUCKET_NAME"]
    bucket_project: str = SETTINGS["GOOGLE_CLOUD_PROJECT"]
    json_creds_path: str = SETTINGS["GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"]

    def get_gcp_bucket(self, bucket_name: str) -> storage.Bucket:
        """Get the GCP bucket object.

        Returns:
            storage.Bucket: GCP bucket object
        """
        try:
            logger.info("Getting GCP bucket")
            storage_client = storage.Client.from_service_account_json(
                json_credentials_path=self.json_creds_path, project=self.bucket_project
            )
            bucket = storage_client.get_bucket(bucket_name)
            logger.info("GCP bucket retrieved")
            return bucket
        except Exception as e:
            logger.error(f"Error getting GCP bucket: {e}")

    def write_model_to_bucket(
        self, bucket_name: str, blob_name: str, model: dict[str, Any]
    ) -> None:
        """Writes a file to the bucket.

        Args:
            blob_name (str): name of blob in bucket
            model (dict[str, Any]): dictionary of model information
        """
        bucket = self.get_gcp_bucket(bucket_name)

        logger.info(f"Creating blob: {blob_name}")
        blob = bucket.blob(blob_name)

        with blob.open("wb") as f:
            joblib.dump(model, f)
        logger.info(f"{model['model']} has been saved to {blob_name}")

    def read_model_from_bucket(
        self, bucket_name: str, blob_name: str
    ) -> pd.DataFrame | None:
        """Reads a file from the bucket.

        Args:
            blob_name (str): name of blob in bucket

        Returns:
            pd.DataFrame | None: dataframe of blob contents or None if blob does not exist
        """
        bucket = self.get_gcp_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return None

        with blob.open("rb") as f:
            return joblib.load(f)

    def read_df_from_bucket(
        self, bucket_name: str, blob_name: str
    ) -> pd.DataFrame | None:
        """Reads a file from the bucket.

        Args:
            blob_name (str): name of blob in bucket

        Returns:
            pd.DataFrame | None: dataframe of blob contents or None if blob does not exist
        """
        bucket = self.get_gcp_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return None

        with blob.open("rb") as f:
            return pd.read_csv(f)

    def write_df_to_bucket(
        self, data: pd.DataFrame, bucket_name: str, blob_name: str
    ) -> None:
        """Writes a file to the bucket.

        Args:
            bucket_name (str): name of bucket
            blob_name (str): name of new blob
            data (pd.DataFrame): dataframe to write to blob
        """
        bucket = self.get_gcp_bucket(bucket_name)

        logger.info(f"Creating blob: {blob_name}")
        blob = bucket.blob(blob_name)

        logger.info(f"Uploading dataframe to {blob_name} as a .csv file")
        blob.upload_from_string(data.to_csv(index=False), "text/csv")


gcp = GCP()
