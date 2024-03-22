"""Provide useful shared tooling."""

import logging
import re

import pandas as pd
import pg8000
from google.cloud import storage
from google.cloud.sql.connector import Connector
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine

logger = logging.getLogger(f"catalystcoop.{__name__}")


class GCSArchive(BaseSettings):
    """Settings for connecting to GCS resources.

    This class looks for several environment variables to configure
    access to GCS resources. These can be set directly, or be in a
    .env file at the top level.

    The following variables need to be set:

    GCS_BUCKET_NAME: Name of bucket where 10k filings are stored.
    GCS_METADATA_DB_INSTANCE_CONNECTION: instance connection string
    in the form 'project:region:instance'.
    GCS_IAM_USER: Email of user of service account trying to connect.
    GCS_METADATA_DB_NAME: Name of DB in instance to connect to.
    """

    model_config = SettingsConfigDict(env_file=".env")

    bucket_name: str = Field(validation_alias="GCS_BUCKET_NAME")
    metadata_db_instance_connection: str = Field(
        validation_alias="GCS_METADATA_DB_INSTANCE_CONNECTION"
    )
    user: str = Field(validation_alias="GCS_IAM_USER")
    metadata_db_name: str = Field(validation_alias="GCS_METADATA_DB_NAME")

    _bucket = PrivateAttr()
    _engine = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize interface to filings archive on GCS."""
        super().__init__(**kwargs)
        self._engine = self._get_engine()
        self._bucket = self._get_bucket()

    def _get_bucket(self):
        """Return cloud storage bucket where SEC10k filings are archived."""
        storage_client = storage.Client()
        return storage_client.bucket(self.bucket_name)

    def _get_engine(self) -> Engine:
        """Initialize a connection pool for a Cloud SQL instance of Postgres.

        Uses the Cloud SQL Python Connector with Automatic IAM Database Authentication.
        """
        # initialize Cloud SQL Python Connector object
        connector = Connector()

        def getconn() -> pg8000.dbapi.Connection:
            conn: pg8000.dbapi.Connection = connector.connect(
                self.metadata_db_instance_connection,
                "pg8000",
                user=self.user,
                db=self.metadata_db_name,
                enable_iam_auth=True,
            )
            return conn

        return create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )

    def get_metadata(self) -> pd:
        """Return dataframe of filing metadata."""
        return pd.read_sql("SELECT * FROM sec10k_metadata", self._engine)

    def get_file(self, year_quarter: str, path: str) -> storage.Blob:
        """Return Blob pointing to file in GCS bucket."""
        return self._bucket.blob(f"sec10k/sec10k-{year_quarter}/{path}")

    def validate_archive(self) -> bool:
        """Validate that all filings described in metadata table exist in GCS bucket."""
        # Get files in archive
        logger.info("Get list of files in archive.")
        archive_filenames = {
            re.sub(r"sec10k/sec10k-\d{4}q\d/", "", blob.name)
            for blob in self._bucket.list_blobs()
        }

        # Get metadata df
        logger.info("Get list of files in metadata.")
        metadata_filenames = set(self.get_metadata()["Filename"])

        if not (valid := archive_filenames == metadata_filenames):
            if len(missing_in_archive := metadata_filenames - archive_filenames) > 0:
                logger.debug(
                    "The following files are listed in metadata, but don't exist"
                    f"in archive: {missing_in_archive}"
                )
            if len(missing_in_metadata := archive_filenames - metadata_filenames) > 0:
                logger.debug(
                    "The following files are listed in archive, but don't exist"
                    f"in metadata: {missing_in_metadata}"
                )
        return valid
