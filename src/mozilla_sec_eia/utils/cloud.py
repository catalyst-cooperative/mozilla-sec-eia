"""Provide useful shared tooling."""

import base64
import io
import logging
import os
import re
from contextlib import contextmanager
from hashlib import md5
from pathlib import Path
from typing import BinaryIO, TextIO

import fitz
import pandas as pd
import pg8000
from google.cloud import secretmanager, storage
from google.cloud.sql.connector import Connector
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from xhtml2pdf import pisa

from mozilla_sec_eia.utils.db_metadata import Base

logger = logging.getLogger(f"catalystcoop.{__name__}")


def _compute_md5(file_path: Path) -> str:
    """Compute an md5 checksum to compare to files in zenodo deposition."""
    hash_md5 = md5()  # noqa: S324
    with Path.open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return base64.b64encode(hash_md5.digest()).decode()


class Exhibit21(BaseModel):
    """This is a class to wrap Exhibit 21's, which are included in many SEC 10ks."""

    ex_21_text: str
    ex_21_version: str
    filename: str

    @classmethod
    def from_10k(cls, filename: str, sec10k_text: str, ex_21_version: str):
        """Extract exhibit 21 from SEC 10K."""
        ex_21_pat = re.compile(
            r"<DOCUMENT>\s?<TYPE>EX-21(\.\d)?([\S\s])*?(</DOCUMENT>)"
        )

        if (match := ex_21_pat.search(sec10k_text)) is None:
            logger.warning(f"Failed to extract exhibit 21 from {filename}")
            return None

        return cls(
            ex_21_text=match.group(0),
            ex_21_version=ex_21_version,
            filename=filename,
        )

    def save_as_pdf(self, file: BinaryIO):
        """Save Exhibit 21 as a PDF in `file`, which can be in memory or on disk."""
        res = pisa.CreatePDF(self.ex_21_text, file)
        if res.err:
            logger.warning(
                f"Failed to create PDF from filing {self.filename} exhibit 21."
            )

    def as_image(self, scale_factor: float = 2.0) -> Image:
        """Return a PIL Image of rendered exhibit 21.

        Args:
            scale_factor: Scale resolutin of output image (higher numbers increase resolution).
        """

        def _v_stack_images(*images):
            """Generate composite of all supplied images."""
            # Get the widest width.
            width = max(image.width for image in images)
            # Add up all the heights.
            height = sum(image.height for image in images)
            composite = Image.new("RGB", (width, height))
            # Paste each image below the one before it.
            y = 0
            for image in images:
                composite.paste(image, (0, y))
                y += image.height
            return composite

        pdf_bytes = io.BytesIO()
        self.save_as_pdf(pdf_bytes)

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [
            Image.open(
                io.BytesIO(
                    doc.load_page(i)
                    .get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
                    .pil_tobytes(format="PNG")
                )
            )
            for i in range(doc.page_count)
        ]
        ex_21_img = _v_stack_images(*pages)

        return ex_21_img


class Sec10K(BaseModel):
    """This is a class to wrap SEC 10K filings."""

    filing_text: str
    filename: str
    cik: int
    year_quarter: str
    ex_21: Exhibit21 | None

    @classmethod
    def from_file(
        cls,
        file: TextIO,
        filename: str,
        cik: int,
        year_quarter: str,
        ex_21_version: str | None,
    ):
        """Cache filing locally, and return class wrapping filing."""
        filing_text = file.read()
        return cls(
            filing_text=filing_text,
            filename=filename,
            cik=cik,
            year_quarter=year_quarter,
            ex_21=(
                Exhibit21.from_10k(filename, filing_text, ex_21_version)
                if ex_21_version
                else None
            ),
        )


class GoogleCloudSettings(BaseSettings):
    """Load environment variables to manage access to cloud resources.

    This class looks for several environment variables to configure
    access to cloud resources. These can be set directly, or be in a
    .env file at the top level.

    The following variables need to be set:

    GCS_FILINGS_BUCKET_NAME: Name of bucket where 10k filings are stored.
    GCS_LABELS_BUCKET_NAME: Name of top-level bucket where labelled training data is stored.
    GCS_METADATA_DB_INSTANCE_CONNECTION: instance connection string
    in the form 'project:region:instance'.
    GCS_IAM_USER: Email of user of service account trying to connect.
    GCS_METADATA_DB_NAME: Name of DB in instance to connect to.
    GCS_PROJECT: Name of google cloud project.
    MLFLOW_TRACKING_URI: URI of mlflow tracking server.
    """

    model_config = SettingsConfigDict(env_file=".env")

    filings_bucket_name: str = Field(validation_alias="GCS_FILINGS_BUCKET_NAME")
    labels_bucket_name: str = Field(validation_alias="GCS_LABELS_BUCKET_NAME")
    metadata_db_instance_connection: str = Field(
        validation_alias="GCS_METADATA_DB_INSTANCE_CONNECTION"
    )
    user: str = Field(validation_alias="GCS_IAM_USER")
    metadata_db_name: str = Field(validation_alias="GCS_METADATA_DB_NAME")
    project: str = Field(validation_alias="GCS_PROJECT")
    tracking_uri: str = Field(validation_alias="MLFLOW_TRACKING_URI")


class GCSArchive(BaseModel):
    """Provides an interface for archived filings on GCS."""

    settings: GoogleCloudSettings = Field(default_factory=lambda: GoogleCloudSettings())

    _filings_bucket = PrivateAttr()
    _labels_bucket = PrivateAttr()
    _engine = PrivateAttr()
    _metadata_df = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        """Initialize interface to filings archive on GCS."""
        super().__init__(**kwargs)
        self._engine = self._get_engine()
        self._filings_bucket = self._get_bucket(self.settings.filings_bucket_name)
        self._labels_bucket = self._get_bucket(self.settings.labels_bucket_name)

        Base.metadata.create_all(self._engine)

    def _get_bucket(self, bucket_name):
        """Return cloud storage bucket where SEC10k filings are archived."""
        storage_client = storage.Client()
        return storage_client.bucket(bucket_name)

    def _get_engine(self) -> Engine:
        """Initialize a connection pool for a Cloud SQL instance of Postgres.

        Uses the Cloud SQL Python Connector with Automatic IAM Database Authentication.
        """
        # initialize Cloud SQL Python Connector object
        connector = Connector()

        def getconn() -> pg8000.dbapi.Connection:
            conn: pg8000.dbapi.Connection = connector.connect(
                self.settings.metadata_db_instance_connection,
                "pg8000",
                user=self.settings.user,
                db=self.settings.metadata_db_name,
                enable_iam_auth=True,
            )
            return conn

        return create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )

    @contextmanager
    def create_session(self) -> Session:
        """Yield sqlalchemy session."""
        with Session(self._engine) as session:
            yield session

    def get_metadata(self) -> pd:
        """Return dataframe of filing metadata."""
        if self._metadata_df is None:
            self._metadata_df = pd.read_sql(
                "SELECT * FROM sec10k_metadata", self._engine
            )
        return self._metadata_df

    def get_filing_blob(self, year_quarter: str, path: str) -> storage.Blob:
        """Return Blob pointing to file in GCS bucket."""
        return self._filings_bucket.blob(f"sec10k/sec10k-{year_quarter}/{path}")

    def _get_local_path(
        self, cache_directory: Path, filing: pd.Series, extension=".html"
    ) -> Path:
        """Return path to a filing in local cache based on metadata."""
        return cache_directory / Path(
            f"{filing['cik']}-{filing['year_quarter']}-"
            f"{filing['filename'].replace('edgar/data/', '').replace('/', '-')}".replace(
                ".txt", extension
            )
        )

    def cache_blob(
        self,
        blob: storage.Blob,
        local_path: Path,
    ) -> Path:
        """Cache a single filing in cache_directory and return path."""
        # Create cache directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if exists := local_path.exists():
            blob.update()
            local_hash = _compute_md5(local_path)
            remote_hash = blob.md5_hash
            refresh = remote_hash != local_hash

        if (not exists) or refresh:
            logger.info(f"Downloading to {local_path}")
            blob.download_to_filename(local_path)
        else:
            logger.info(f"{local_path} is already cached")

        return local_path

    def get_filings(
        self,
        filing_selection: pd.DataFrame,
        cache_directory: Path = Path("./sec10k_filings"),
    ) -> list[Sec10K]:
        """Caches filings locally for quick access, and list of Sec10K objects.

        Args:
            filing_selection: Pandas dataframe with same schema as metadata df where each row
                is a filing to return.
            cache_directory: Path to directory where filings should be cached.
        """
        filings = []
        for _, filing in filing_selection.iterrows():
            blob = self.get_filing_blob(filing["year_quarter"], filing["filename"])
            local_path = self._get_local_path(cache_directory, filing)
            filing_path = self.cache_blob(blob, local_path)

            with filing_path.open() as f:
                filings.append(
                    Sec10K.from_file(
                        file=f,
                        filename=filing["filename"],
                        cik=filing["cik"],
                        year_quarter=filing["year_quarter"],
                        ex_21_version=filing["exhibit_21_version"],
                    )
                )
        return filings

    def iterate_filings(
        self,
        filing_selection: pd.DataFrame,
        cache_directory: Path = Path("./sec10k_filings"),
    ):
        """Iterate through filings without caching locally.

        This method will only download a filing when ``next()`` is called on the
        returned iterator, and it will not save filings to disk. This is useful
        when working with a large selection of filings to avoid using excessive
        amounts of disk space to save all filings.

        Args:
            filing_selection: Pandas dataframe with same schema as metadata df where each row
                is a filing to return.
            cache_directory: Path to directory where filings should be cached.
        """
        for _, filing in filing_selection.iterrows():
            yield Sec10K.from_file(
                file=io.StringIO(
                    self.get_filing_blob(
                        filing["year_quarter"], filing["filename"]
                    ).download_as_text()
                ),
                filename=filing["filename"],
                cik=filing["cik"],
                year_quarter=filing["year_quarter"],
                ex_21_version=filing["exhibit_21_version"],
            )

    def cache_training_data(
        self, json_cache_path: Path, pdf_cache_path: Path, overwrite_pdfs: bool = False
    ):
        """Cache labeled training data stored on GCS for local use."""
        json_cache_path.mkdir(parents=True, exist_ok=True)
        pdf_cache_path.mkdir(parents=True, exist_ok=True)
        metadata_df = self.get_metadata()
        label_name_pattern = re.compile(r"(\d+)-\d{4}q[1-4]-\d+-(.+)")
        for blob in self._labels_bucket.list_blobs(match_glob="labeled/*"):
            if blob.name == "labeled/":
                continue

            # Cache labels
            self.cache_blob(blob, json_cache_path / blob.name.replace("labeled/", ""))

            # Cache filing
            match = label_name_pattern.search(blob.name)
            filename = f"edgar/data/{match.group(1)}/{match.group(2)}.txt"
            filing_metadata = metadata_df[metadata_df["filename"] == filename]
            filing = self.get_filings(filing_metadata)[0]
            pdf_path = self._get_local_path(
                pdf_cache_path, filing_metadata.iloc[0], extension=".pdf"
            )
            if not pdf_path.exists() or overwrite_pdfs:
                with pdf_path.open("wb") as f:
                    filing.ex_21.save_as_pdf(f)

    def validate_archive(self) -> bool:
        """Validate that all filings described in metadata table exist in GCS bucket."""
        # Get files in archive
        logger.info("Get list of files in archive.")
        archive_filenames = {
            re.sub(r"sec10k/sec10k-\d{4}q\d/", "", blob.name)
            for blob in self._filings_bucket.list_blobs()
        }

        # Get metadata df
        logger.info("Get list of files in metadata.")
        metadata_filenames = set(self.get_metadata()["filename"])

        if not (valid := archive_filenames == metadata_filenames):
            logger.warning("Archive validation failed.")
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
        else:
            logger.info("Archive is valid!")
        return valid


def _access_secret_version(secret_id: str, project_id: str, version_id="latest"):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the decoded payload.
    return response.payload.data.decode("UTF-8")


def initialize_mlflow():
    """Set appropriate environment variables to prepare connection to tracking server."""
    settings = GoogleCloudSettings()
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _access_secret_version(
        "mlflow_admin_password", settings.project
    )
    os.environ["MLFLOW_TRACKING_URI"] = settings.tracking_uri
    os.environ["MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE"] = "20971520"
    os.environ["MLFLOW_GCS_UPLOAD_CHUNK_SIZE"] = "20971520"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "900"
    logger.info(f"Initialized tracking with mlflow server: {settings.tracking_uri}")
