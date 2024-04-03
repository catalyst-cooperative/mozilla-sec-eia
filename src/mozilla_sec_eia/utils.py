"""Provide useful shared tooling."""

import base64
import io
import logging
import re
from hashlib import md5
from pathlib import Path
from typing import BinaryIO

import fitz
import pandas as pd
import pg8000
from google.cloud import storage
from google.cloud.sql.connector import Connector
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine
from xhtml2pdf import pisa

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

    def save_as_pdf(self, file: BinaryIO):
        """Save Exhibit 21 as a PDF in `file`, which can be in memory or on disk."""
        res = pisa.CreatePDF(self.ex_21_text, file)
        if res.err:
            logger.warning(
                f"Failed to create PDF from filing {self.filename} exhibit 21."
            )

    def as_image(self) -> Image:
        """Return a PIL Image of rendered exhibit 21."""

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
                io.BytesIO(doc.load_page(i).get_pixmap().pil_tobytes(format="PNG"))
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
    ex_21_version: str | None

    @classmethod
    def from_path(
        cls, filing_path: Path, cik: int, year_quarter: str, ex_21_version: str | None
    ):
        """Cache filing locally, and return class wrapping filing."""
        with filing_path.open() as f:
            return cls(
                filing_text=f.read(),
                filename=filing_path.name,
                cik=cik,
                year_quarter=year_quarter,
                ex_21_version=ex_21_version,
            )

    def get_ex_21(self) -> Exhibit21 | None:
        """Return EX 21 if filing has one."""
        if self.ex_21_version is not None:
            ex_21_pat = re.compile(
                r"<DOCUMENT>\s?<TYPE>EX-21(\.\d)?([\S\s])*?(</DOCUMENT>)"
            )

            if ex_21_pat is None:
                logger.warning(f"Failed to extract exhibit 21 from {self.filename}")
                return None

            return Exhibit21(
                ex_21_text=ex_21_pat.search(self.filing_text).group(0),
                ex_21_version=self.ex_21_version,
                filename=self.filename,
            )

        logger.warning(f"{self.filename} does not contain an exhibit 21")
        return None


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
    _metadata_df = PrivateAttr(default=None)

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
        if self._metadata_df is None:
            self._metadata_df = pd.read_sql(
                "SELECT * FROM sec10k_metadata", self._engine
            )
        return self._metadata_df

    def get_blob(self, year_quarter: str, path: str) -> storage.Blob:
        """Return Blob pointing to file in GCS bucket."""
        return self._bucket.blob(f"sec10k/sec10k-{year_quarter}/{path}")

    def _get_local_path(self, cache_directory: Path, filing: pd.Series) -> Path:
        """Return path to a filing in local cache based on metadata."""
        return cache_directory / Path(
            f"{filing['CIK']}-{filing['year_quarter']}-"
            f"{filing['Filename'].replace('edgar/data/', '').replace('/', '-')}"
        )

    def cache_filing(
        self,
        filing: pd.Series,
        cache_directory: Path = Path("./sec10k_filings"),
    ) -> Path:
        """Cache a single filing in cache_directory and return path."""
        # Create cache directory
        cache_directory.mkdir(parents=True, exist_ok=True)

        blob = self.get_blob(filing["year_quarter"], filing["Filename"])
        local_path = self._get_local_path(cache_directory, filing)

        if exists := local_path.exists():
            blob.update()
            local_hash = _compute_md5(local_path)
            remote_hash = blob.md5_hash
            refresh = remote_hash != local_hash

        if (not exists) or refresh:
            logger.info(f"Downloading {filing['Filename']}")
            blob.download_to_filename(local_path)
        else:
            logger.info(f"{filing['Filename']} is already cached")

        return local_path

    def get_filings(
        self,
        filing_selection: pd.DataFrame,
        cache_directory: Path = Path("./sec10k_filings"),
    ) -> list[Sec10K]:
        """Caches filings locally for quick access, and returns generator producing paths.

        Args:
            filing_selection: Pandas dataframe with same schema as metadata df where each row
                is a filing to return.
            cache_directory: Path to directory where filings should be cached.
        """
        filings = []
        for _, filing in filing_selection.iterrows():
            filing_path = self.cache_filing(filing, cache_directory)
            filings.append(
                Sec10K.from_path(
                    filing_path,
                    filing["CIK"],
                    filing["year_quarter"],
                    filing["exhibit_21_version"],
                )
            )
        return filings

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
