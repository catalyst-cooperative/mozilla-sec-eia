"""Create inference dataset for exhibit 21 extraction model."""

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

from ...utils.cloud import GCSArchive
from ...utils.pdf import get_image_dict, get_pdf_data_from_path
from .common import BBOX_COLS_PDF, format_label_studio_output, normalize_bboxes

logger = logging.getLogger(f"catalystcoop.{__name__}")


def format_unlabeled_pdf_dataframe(pdfs_dir: Path):
    """Read and format PDFs into a dataframe (without labels)."""
    inference_df = pd.DataFrame()
    for pdf_filename in os.listdir(pdfs_dir):
        if not pdf_filename.endswith(".pdf"):
            continue
        src_path = pdfs_dir / pdf_filename
        filename = Path(pdf_filename).stem
        extracted, pg = get_pdf_data_from_path(src_path)
        txt = extracted["pdf_text"]
        pg_meta = extracted["page"]
        # normalize bboxes between 0 and 1000 for Hugging Face
        txt = normalize_bboxes(txt_df=txt, pg_meta_df=pg_meta)
        txt.loc[:, "id"] = filename
        inference_df = pd.concat([inference_df, txt])
    return inference_df


def _cache_pdfs(
    filings: pd.DataFrame, cloud_interface: GCSArchive, pdf_dir: Path
) -> pd.DataFrame:
    """Iterate filings and cache pdfs."""
    extraction_metadata = pd.DataFrame(
        {
            "filename": pd.Series(dtype=str),
            "success": pd.Series(dtype=bool),
            "notes": pd.Series(dtype=str),
        }
    ).set_index("filename")

    for filing in cloud_interface.iterate_filings(filings):
        pdf_path = cloud_interface.get_local_filename(
            cache_directory=pdf_dir, filing=filing, extension=".pdf"
        )

        # Some filings are poorly formatted and fail in `save_as_pdf`
        # We want a record of these but don't want to stop run
        try:
            with pdf_path.open("wb") as f:
                filing.ex_21.save_as_pdf(f)
        except Exception as e:
            extraction_metadata.loc[filing.filename, ["success"]] = False
            extraction_metadata.loc[filing.filename, ["note"]] = str(e)

        # Some pdfs are empty. Check for these and remove from dir
        if pdf_path.stat().st_size == 0:
            extraction_metadata.loc[filing.filename, ["success"]] = False
            extraction_metadata.loc[filing.filename, ["note"]] = "PDF empty"
            pdf_path.unlink()

    return extraction_metadata


def create_inference_dataset(
    filing_metadata: pd.DataFrame, cloud_interface: GCSArchive, has_labels: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a Hugging Face Dataset from PDFs for inference."""
    filings_with_ex21 = filing_metadata[~filing_metadata["exhibit_21_version"].isna()]

    # Parse PDFS
    with (
        tempfile.TemporaryDirectory() as pdfs_dir,
        tempfile.TemporaryDirectory() as labeled_json_dir,
    ):
        pdfs_dir = Path(pdfs_dir)
        labeled_json_dir = Path(labeled_json_dir)

        extraction_metadata = _cache_pdfs(
            filings_with_ex21,
            cloud_interface=cloud_interface,
            pdf_dir=pdfs_dir,
        )
        if has_labels:
            inference_df = format_label_studio_output(
                labeled_json_dir=labeled_json_dir, pdfs_dir=pdfs_dir
            )
        else:
            inference_df = format_unlabeled_pdf_dataframe(pdfs_dir=pdfs_dir)
        image_dict = get_image_dict(pdfs_dir)

    annotations = []
    for filename, image in image_dict.items():
        annotation = {
            "id": filename,
            "tokens": inference_df.groupby("id")["text"].apply(list).loc[filename],
            "bboxes": inference_df.loc[inference_df["id"] == filename, :][BBOX_COLS_PDF]
            .to_numpy()
            .tolist(),
            "image": image.tobytes(),
            "mode": image.mode,
            "width": image.size[0],
            "height": image.size[1],
        }
        if has_labels:
            annotation["ner_tags"] = (
                inference_df.groupby("id")["ner_tag"].apply(list).loc[filename]
            )
        annotations.append(annotation)

    return extraction_metadata, pd.DataFrame(annotations)
