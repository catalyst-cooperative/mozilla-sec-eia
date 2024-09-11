"""Provide useful shared tooling."""

import base64
import io
import logging
import re
from contextlib import contextmanager
from hashlib import md5
from pathlib import Path
from typing import BinaryIO, TextIO

import fitz
import pandas as pd
import pg8000
from dagster import ConfigurableResource, EnvVar
from google.cloud import storage
from google.cloud.sql.connector import Connector
from PIL import Image
from pydantic import BaseModel, PrivateAttr
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from xhtml2pdf import pisa

from .db_metadata import Base, Sec10kMetadata

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


class GCSArchive(ConfigurableResource):
    """Provides an interface for archived filings on GCS.

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

    filings_bucket_name: str
    labels_bucket_name: str
    metadata_db_instance_connection: str
    user: str
    metadata_db_name: str
    project: str

    _filings_bucket = PrivateAttr()
    _labels_bucket = PrivateAttr()
    _engine = PrivateAttr()

    def setup_for_execution(self, context):
        """Initialize interface to filings archive on GCS."""
        self._engine = self._get_engine()
        self._filings_bucket = self._get_bucket(self.filings_bucket_name)
        self._labels_bucket = self._get_bucket(self.labels_bucket_name)

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

    @contextmanager
    def create_session(self) -> Session:
        """Yield sqlalchemy session."""
        with Session(self._engine) as session:
            yield session

    def get_metadata(self, year_quarter: str | None = None) -> pd:
        """Return dataframe of filing metadata."""
        selection = select(Sec10kMetadata)
        if year_quarter is not None:
            selection = selection.where(Sec10kMetadata.year_quarter == year_quarter)

        return pd.read_sql(selection, self._engine)

    def get_filing_blob(self, year_quarter: str, path: str) -> storage.Blob:
        """Return Blob pointing to file in GCS bucket."""
        return self._filings_bucket.blob(f"sec10k/sec10k-{year_quarter}/{path}")

    def get_local_filename(
        self, cache_directory: Path, filing: pd.Series, extension=".html"
    ) -> Path:
        """Return path to a filing in local cache based on metadata."""
        return cache_directory / Path(
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
        cache_pdf: bool = False,
    ) -> list[Sec10K]:
        """Get list of Sec10K objects and cache filings.

        Args:
            filing_selection: Pandas dataframe with same schema as metadata df where each row
                is a filing to return.
            cache_directory: Path to directory where filings should be cached.
            cache_pdf: Boolean indicating whether to also cache a PDF of the Ex. 21
        """
        filings = []
        for _, filing in filing_selection.iterrows():
            blob = self.get_filing_blob(filing["year_quarter"], filing["filename"])
            local_path = self.get_local_filename(cache_directory, filing)
            filing_path = self.cache_blob(blob, local_path)

            with filing_path.open() as f:
                sec10k_filing = Sec10K.from_file(
                    file=f,
                    filename=filing["filename"],
                    cik=filing["cik"],
                    year_quarter=filing["year_quarter"],
                    ex_21_version=filing["exhibit_21_version"],
                )
                filings.append(sec10k_filing)
            if cache_pdf:
                pdf_path = self.get_local_filename(
                    cache_directory=cache_directory,
                    filing=filing,
                    extension=".pdf",
                )
                if not pdf_path.exists():
                    with pdf_path.open("wb") as f:
                        sec10k_filing.ex_21.save_as_pdf(f)
        return filings

    def iterate_filings(
        self,
        filing_selection: pd.DataFrame,
    ):
        """Iterate through filings without caching locally.

        This method will only download a filing when ``next()`` is called on the
        returned iterator, and it will not save filings to disk. This is useful
        when working with a large selection of filings to avoid using excessive
        amounts of disk space to save all filings.

        Args:
            filing_selection: Pandas dataframe with same schema as metadata df where each row
                is a filing to return.
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
        self,
        json_cache_path: Path,
        pdf_cache_path: Path,
        gcs_folder_name: str = "labeled/",
        overwrite_pdfs: bool = False,
    ):
        """Cache labeled training data stored on GCS for local use."""
        json_cache_path.mkdir(parents=True, exist_ok=True)
        pdf_cache_path.mkdir(parents=True, exist_ok=True)
        metadata_df = self.get_metadata()
        label_name_pattern = re.compile(r"(\d+)-\d{4}q[1-4]-\d+-(.+)")
        if gcs_folder_name[-1] != "/":
            gcs_folder_name += "/"
        for blob in self._labels_bucket.list_blobs(match_glob=f"{gcs_folder_name}*"):
            if blob.name == gcs_folder_name:
                continue

            # Cache labels
            self.cache_blob(
                blob, json_cache_path / blob.name.replace(gcs_folder_name, "")
            )

            # Cache filing
            match = label_name_pattern.search(blob.name)
            filename = f"edgar/data/{match.group(1)}/{match.group(2)}.txt"
            filing_metadata = metadata_df[metadata_df["filename"] == filename]
            filing = self.get_filings(filing_metadata)[0]
            pdf_path = self.get_local_filename(
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


def get_metadata_filename(local_filename: str):
    """Transform a local filename into the filename in GCSArchiver metadata."""
    return "edgar/data/" + local_filename.replace("-", "/", 1) + ".txt"


cloud_interface_resource = GCSArchive(
    filings_bucket_name=EnvVar("GCS_FILINGS_BUCKET_NAME"),
    labels_bucket_name=EnvVar("GCS_LABELS_BUCKET_NAME"),
    metadata_db_instance_connection=EnvVar("GCS_METADATA_DB_INSTANCE_CONNECTION"),
    user=EnvVar("GCS_IAM_USER"),
    metadata_db_name=EnvVar("GCS_METADATA_DB_NAME"),
    project=EnvVar("GCS_PROJECT"),
)