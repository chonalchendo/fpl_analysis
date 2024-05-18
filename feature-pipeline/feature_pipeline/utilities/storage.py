from google.cloud import storage
from dataclasses import dataclass
from typing import Any
import pandas as pd
import joblib
from feature_pipeline.utilities.utils import get_logger
from feature_pipeline.core.settings import SETTINGS

logger = get_logger(__name__)


@dataclass
class GCP:
    """Dataclass for interacting with Google Cloud Platform."""

    bucket_project: str = SETTINGS["GOOGLE_CLOUD_PROJECT"]
    json_creds_path: str = SETTINGS["GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"]

    def create_gcp_bucket(self, bucket_name: str) -> None:
        """Creates a GCP bucket.

        Args:
            bucket_name (str): name of bucket to create
        """
        try:
            logger.info("Creating GCP bucket")
            storage_client = storage.Client.from_service_account_json(
                json_credentials_path=self.json_creds_path, project=self.bucket_project
            )
            bucket = storage_client.create_bucket(bucket_name)
            logger.info(f"GCP bucket {bucket.name} created")
        except Exception as e:
            logger.error(f"Error creating GCP bucket: {e}")

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

    def write_blob_to_bucket(
        self, bucket_name: str, blob_name: str, data: pd.DataFrame
    ) -> None:
        """Writes a file to the bucket.

        Args:
            blob_name (str): name of blob in bucket
            model (dict[str, Any]): dictionary of model information
        """
        bucket = self.get_gcp_bucket(bucket_name)

        # if bucket is None:
        #     self.create_gcp_bucket(bucket_name)
        #     bucket = self.get_gcp_bucket(bucket_name)

        logger.info(f"Creating blob: {blob_name}")
        blob = bucket.blob(blob_name)

        blob.upload_from_string(data.to_csv(index=False), "text/csv")

    def read_blob_from_bucket(
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
            return pd.read_pickle(f)


gcp = GCP()
