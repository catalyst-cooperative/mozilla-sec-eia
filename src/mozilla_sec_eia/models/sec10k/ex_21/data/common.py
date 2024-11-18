"""Implement methods used to construct both inference and training sets."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mozilla_sec_eia.library import validation_helpers

from ...utils.pdf import get_pdf_data_from_path

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

BBOX_COLS_PDF = [
    "top_left_x_pdf",
    "top_left_y_pdf",
    "bottom_right_x_pdf",
    "bottom_right_y_pdf",
]


def normalize_bboxes(txt_df, pg_meta_df):
    """Normalize bboxes between 0 and 1000."""
    txt_df["top_left_x_pdf"] = (
        txt_df["top_left_x_pdf"] / pg_meta_df.width_pdf_coord.iloc[0] * 1000
    )
    txt_df["top_left_y_pdf"] = (
        txt_df["top_left_y_pdf"] / pg_meta_df.height_pdf_coord.iloc[0] * 1000
    )
    txt_df["bottom_right_x_pdf"] = (
        txt_df["bottom_right_x_pdf"] / pg_meta_df.width_pdf_coord.iloc[0] * 1000
    )
    txt_df["bottom_right_y_pdf"] = (
        txt_df["bottom_right_y_pdf"] / pg_meta_df.height_pdf_coord.iloc[0] * 1000
    )
    return txt_df


def unnormalize_box(bbox, width, height):
    """Unnormalize bboxes for drawing onto an image."""
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def get_id_label_conversions(labels):
    """Return dicts mapping ids to labels and labels to ids."""
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in enumerate(labels)}
    return id2label, label2id


def iob_to_label(label):
    """Convert an IOB entity label to a standard string label.

    i.e. 'B-Subsidiary' becomes 'Subsidiary'.
    """
    label = label[2:]
    if not label:
        return "other"
    return label


def _is_cik_in_training_data(labeled_json_filename, tracking_df):
    cik = int(labeled_json_filename.split("/")[-1].split("-")[0])
    return cik in tracking_df.CIK.unique()


def format_label_studio_output(
    labeled_json_dir: Path,
    pdfs_dir: Path,
) -> pd.DataFrame:
    """Format Label Studio output JSONs into dataframe."""
    labeled_df = pd.DataFrame()
    tracking_df = validation_helpers.load_training_data("ex21_labeled_filings.csv")

    for json_filename in os.listdir(labeled_json_dir):
        if not json_filename[0].isdigit() or json_filename.endswith(".json"):
            continue
        json_file_path = labeled_json_dir / json_filename
        with Path.open(json_file_path) as j:
            doc_dict = json.loads(j.read())
            filename = doc_dict["task"]["data"]["ocr"].split("/")[-1].split(".")[0]
            # check if old local naming schema is being used
            if len(filename.split("-")) == 6:
                filename = "-".join(filename.split("-")[2:])
            if not _is_cik_in_training_data(filename, tracking_df=tracking_df):
                continue
            pdf_filename = filename + ".pdf"
            src_path = pdfs_dir / pdf_filename
            extracted, pg = get_pdf_data_from_path(src_path)
            txt = extracted["pdf_text"]
            pg_meta = extracted["page"]
            # normalize bboxes between 0 and 1000 for Hugging Face
            txt = normalize_bboxes(txt_df=txt, pg_meta_df=pg_meta)
            # parse the output dictionary of labeled bounding boxes from Label Studio
            doc_df = pd.DataFrame()
            for item in doc_dict["result"]:
                value = item["value"]
                # sometimes Label Studio will fill in an empty list as a label
                # when there is really no label
                # TODO: do this without dict comprehension?
                if ("labels" in value) and value["labels"] == []:
                    value = {k: v for k, v in value.items() if k != "labels"}
                ind = int(item["id"].split("_")[-1])
                doc_df = pd.concat([doc_df, pd.DataFrame(value, index=[ind])])
            # combine the bounding boxes for each word
            doc_df = doc_df.groupby(level=0).first()
            txt.loc[:, "id"] = filename
            # TODO: probably want to filter out these empty Ex. 21 docs
            # the doc might not have any labels in it if it was an empty Ex. 21
            if "labels" not in doc_df:
                doc_df.loc[:, "labels"] = pd.Series()
            output_df = pd.concat([txt, doc_df[["labels"]]], axis=1)
            labeled_df = pd.concat([labeled_df, output_df])

    # fill in unlabeled words and clean up labeled dataframe
    labeled_df["labels"] = labeled_df["labels"].fillna("O")
    labeled_df = labeled_df.rename(columns={"labels": "ner_tag"})
    non_id_columns = [col for col in labeled_df.columns if col != "id"]
    labeled_df = labeled_df.loc[:, ["id"] + non_id_columns]

    # TODO: add in sanity checks on labeled_df bounding boxes to make sure
    # that no value is above 1000 or below 0

    return labeled_df


def _sort_by_label_priority(target_array):
    _, label2id = get_id_label_conversions(LABELS)
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
