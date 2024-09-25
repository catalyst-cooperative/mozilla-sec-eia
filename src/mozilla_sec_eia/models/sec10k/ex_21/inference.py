"""Module for formatting inputs and performing inference with a fine-tuned LayoutLM model."""

import logging
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel

from ..entities import Ex21CompanyOwnership
from ..utils.cloud import GCSArchive
from ..utils.layoutlm import (
    normalize_bboxes,
)
from ..utils.pdf import (
    get_pdf_data_from_path,
)
from .create_labeled_dataset import (
    BBOX_COLS_PDF,
    format_label_studio_output,
    get_image_dict,
)

# When handling multi page documents LayoutLM uses a sliding 'frame'
# with some overlap between frames. The overlap creates multiple
# predictions for the same bounding boxes. If there are multiple mode
# predictions for a bounding box, then ties are broken by setting
# a priority for the labels and choosing the highest priority label.
LABEL_PRIORITY = [
    "I-Subsidiary",
    "I-Loc",
    "I-Own_Per",
    "B-Subsidiary",
    "B-Loc",
    "B-Own_Per",
    "O",
]

LABELS = [
    "O",
    "B-Subsidiary",
    "I-Subsidiary",
    "B-Loc",
    "I-Loc",
    "B-Own_Per",
    "I-Own_Per",
]

BBOX_COLS = ["top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"]
label2id = {v: k for k, v in enumerate(LABELS)}

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


def extract_filings(
    filings: pd.DataFrame,
    layoutlm: PyFuncModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create huggingface dataset from filings and perform extraction."""
    try:
        failed_metadata, dataset = create_inference_dataset(
            filing_metadata=filings,
            cloud_interface=GCSArchive(),
            has_labels=False,
        )
        metadata, extracted = layoutlm.predict(dataset)
        metadata = pd.concat([failed_metadata, metadata])
    except Exception as e:
        logger.warning(traceback.format_exc())
        logger.warning(f"Error while extracting filings: {filings.index}")
        metadata = pd.DataFrame(
            {
                "filename": filings.index,
                "success": [False] * len(filings),
                "notes": [str(e)] * len(filings),
            }
        ).set_index("filename")
        extracted = Ex21CompanyOwnership.example(size=0)
    return metadata, extracted


def _sort_by_label_priority(target_array):
    id_priority = [label2id[label] for label in LABEL_PRIORITY]
    # Create a priority map from the label priority
    priority_map = {val: idx for idx, val in enumerate(id_priority)}
    # Sort the target array based on the priority map
    sorted_array = sorted(target_array, key=lambda x: priority_map.get(x, float("inf")))
    return sorted_array


def get_flattened_mode_predictions(token_boxes_tensor, predictions_tensor):
    """Get the mode prediction for each box in an Ex. 21.

    When handling multi page documents LayoutLM uses a sliding 'frame'
    with some overlap between frames. The overlap creates multiple
    predictions for the same bounding boxes. Thus it's necessary to find
    the mode of all the predictions for a bounding box and use that as the
    single prediction for each box. If there are multiple mode
    predictions for a bounding box, then ties are broken by setting
    a priority for the labels (LABEL_PRIORITY) and choosing the highest priority
    label.
    """
    # Flatten the tensors
    flat_token_boxes = token_boxes_tensor.view(-1, 4)
    flat_predictions = predictions_tensor.view(-1)

    boxes = flat_token_boxes.numpy()
    predictions = flat_predictions.numpy()

    # Find unique boxes and indices
    unique_boxes, inverse_indices = np.unique(boxes, axis=0, return_inverse=True)

    # Compute the mode for each unique bounding box
    # for each unique box in boxes, create a list with all predictions for that box
    # get the indices in predictions where the corresponding index in boxes is
    unique_box_predictions = [
        predictions[np.where(inverse_indices == i)[0]] for i in range(len(unique_boxes))
    ]
    pred_counts = [np.bincount(arr) for arr in unique_box_predictions]
    # Compute the mode of predictions for each group
    # break ties by taking into account LABEL_PRIORITY
    modes = np.array(
        [
            _sort_by_label_priority(np.where(arr == np.max(arr))[0])[0]
            for arr in pred_counts
        ]
    )
    flattened_modes = modes[inverse_indices]

    return flattened_modes
