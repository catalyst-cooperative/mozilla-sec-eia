"""Provide useful shared tooling."""

import base64
import io
import logging
import re
import typing
from hashlib import md5
from pathlib import Path

import fitz
import pandas as pd
import pg8000
from google.cloud import storage
from google.cloud.sql.connector import Connector
from PIL import Image
from pydantic import Field, PrivateAttr
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

    def _v_stack_images(self, *images):
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

    def get_filing(
        self,
        filing: pd.Series,
        cache_directory: Path = Path("./sec10k_filings"),
    ) -> Path:
        """Cache a single filing and return path."""
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
            blob.download_to_filename(local_path)

        return local_path

    def get_filings(
        self,
        filing_selection: pd.DataFrame,
        cache_directory: Path = Path("./sec10k_filings"),
    ) -> typing.Generator[Path, None, None]:
        """Caches filings locally for quick access, and returns generator producing paths.

        Args:
            filing_selection: Pandas dataframe with same schema as metadata df where each row
                is a filing to return.
            cache_directory: Path to directory where filings should be cached.
        """
        for _, filing in filing_selection.iterrows():
            yield self.get_filing(filing, cache_directory)

    def get_ex_21_from_filing(
        self,
        filing: Path,
    ) -> Image:
        """Extract exhibit 21 from raw filing and return PIL Image of table."""
        ex_21_pat = re.compile(
            r"<DOCUMENT>\s?<TYPE>EX-21(\.\d)?([\S\s])*?(</DOCUMENT>)"
        )
        with filing.open() as f:
            ex_21_text = ex_21_pat.search(f.read()).group(0)

        pdf_bytes = io.BytesIO()
        res = pisa.CreatePDF(ex_21_text, pdf_bytes)
        if res.err:
            raise RuntimeError(
                f"Failed to create PDF from filing {str(filing)} exhibit 21."
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [
            Image.open(
                io.BytesIO(doc.load_page(i).get_pixmap().pil_tobytes(format="PNG"))
            )
            for i in range(doc.page_count)
        ]
        ex_21_img = self._v_stack_images(*pages)

        # Crop white border
        nonwhite_positions = [
            (x, y)
            for x in range(ex_21_img.size[0])
            for y in range(ex_21_img.size[1])
            if ex_21_img.getdata()[x + y * ex_21_img.size[0]] != (255, 255, 255)
        ]
        rect = (
            min([x for x, y in nonwhite_positions]),
            min([y for x, y in nonwhite_positions]),
            max([x for x, y in nonwhite_positions]),
            max([y for x, y in nonwhite_positions]),
        )
        return ex_21_img.crop(rect)

    def get_ex_21_images(
        self,
        filing_selection: pd.DataFrame,
        cache_directory: Path = Path("./sec10k_filings"),
    ) -> typing.Generator[Image, None, None]:
        """Generator that returns image of exhibit 21 from filings."""
        for filing_path in self.get_filings(filing_selection, cache_directory):
            yield self.get_ex_21_from_filing(filing_path)

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
