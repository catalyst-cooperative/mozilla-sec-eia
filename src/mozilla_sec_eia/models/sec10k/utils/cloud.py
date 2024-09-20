"""Provide useful shared tooling."""

import base64
import io
import logging
import re
from hashlib import md5
from pathlib import Path
from typing import BinaryIO, TextIO

import fitz
import pandas as pd
from dagster import ConfigurableResource
from PIL import Image
from pydantic import BaseModel
from upath import UPath
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
        # TODO: probably should make a "corrections" file that has CSS/HTML replacements
        # to make PDF render
        # TODO: should probably also catch errors and not fail
        if "border-bottom: black thin solid;" in self.ex_21_text:
            self.ex_21_text = self.ex_21_text.replace(
                "border-bottom: black thin solid;", "border-bottom: 1px solid black;"
            )
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
    """Provides an interface for archived filings on GCS."""

    filings_bucket: str = "gs://2de2b9f52c99a240-bucket-sec-10ks/"
    labels_bucket: str = "gs://labeled-ex21-filings/"
    outputs_bucket: str = "gs://sec10k-outputs/"

    @property
    def filings_bucket_path(self):
        """Return UPath of filings bucket."""
        return UPath(self.filings_bucket)

    @property
    def labels_bucket_path(self):
        """Return UPath of filings bucket."""
        return UPath(self.labels_bucket)

    @property
    def outputs_bucket_path(self):
        """Return UPath of filings bucket."""
        return UPath(self.outputs_bucket)

    def get_metadata(self, year_quarter: str | None = None) -> pd:
        """Return dataframe of filing metadata."""
        selection = None
        if year_quarter is not None:
            selection = ["year_quarter", "==", year_quarter]

        return pd.read_parquet(
            self.outputs_bucket_path / "sec10k_filing_metadata", filters=selection
        )

    def get_local_filename(
        self, cache_directory: Path, filing: pd.Series | Sec10K, extension=".html"
    ) -> Path:
        """Return path to a filing in local cache based on metadata."""
        if isinstance(filing, pd.Series):
            filename = filing["filename"]
        else:
            filename = filing.filename
        return cache_directory / Path(
            f"{filename.replace('edgar/data/', '').replace('/', '-')}".replace(
                ".txt", extension
            )
        )

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
            local_path = self.get_local_filename(cache_directory, filing)
            year_quarter = filing["year_quarter"]
            if not local_path.exists():
                with local_path.open("w") as f:
                    f.write(
                        (
                            self.filings_bucket_path
                            / "sec10k"
                            / f"sec10k-{year_quarter}"
                            / filing.filename
                        ).read_text()
                    )

            with local_path.open() as f:
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
        for filename, filing in filing_selection.iterrows():
            filepath = f"sec10k/sec10k-{filing.year_quarter}/{filename}"
            yield Sec10K.from_file(
                file=io.StringIO((self.filings_bucket_path / filepath).read_text()),
                filename=filename,
                cik=filing["cik"],
                year_quarter=filing["year_quarter"],
                ex_21_version=filing["exhibit_21_version"],
            )

    def cache_training_data(
        self,
        json_cache_path: Path,
        pdf_cache_path: Path,
        gcs_folder_name: str = "labeledv0.2",
        overwrite_pdfs: bool = False,
    ):
        """Cache labeled training data stored on GCS for local use."""
        json_cache_path.mkdir(parents=True, exist_ok=True)
        pdf_cache_path.mkdir(parents=True, exist_ok=True)
        metadata_df = self.get_metadata()
        # label_name_pattern = re.compile(r"(\d+)-\d{4}q[1-4]-\d+-(.+)")
        label_name_pattern = re.compile(r"(\d+)-(.+)")
        # Cache filings and labels
        filenames = []
        direc = self.labels_bucket_path / gcs_folder_name
        logger.info(direc.is_dir())
        for file in direc.iterdir():
            if file.name == gcs_folder_name:
                continue
            # Cache labels
            with (json_cache_path / file.name).open("w") as f:
                f.write(file.read_text())

            # Cache filing
            match = label_name_pattern.search(file.name)
            filenames.append(f"edgar/data/{match.group(1)}/{match.group(2)}.txt")

        metadata_df = metadata_df.reset_index()
        filings = metadata_df[metadata_df["filename"].isin(filenames)]
        self.get_filings(
            filing_selection=filings,
            cache_directory=pdf_cache_path,
            cache_pdf=True,
        )

    def validate_archive(self) -> bool:
        """Validate that all filings described in metadata table exist in GCS bucket."""
        # Get files in archive
        logger.info("Get list of files in archive.")
        archive_filenames = {
            filing.name for filing in self.filings_bucket_path.iterdir()
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


cloud_interface_resource = GCSArchive()
