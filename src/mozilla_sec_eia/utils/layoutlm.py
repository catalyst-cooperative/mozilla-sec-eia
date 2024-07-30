"""Util functions for training and predicting with LayoutLM on Ex. 21 tables."""

import mlflow
from PIL import ImageDraw, ImageFont
from transformers import (
    Trainer,
)

from mozilla_sec_eia.utils.cloud import initialize_mlflow


def log_model(finetuned_model: Trainer):
    """Log fine-tuned model to mlflow artifacts."""
    model = {"model": finetuned_model.model, "tokenizer": finetuned_model.tokenizer}
    mlflow.transformers.log_model(
        model, artifact_path="layoutlm_extractor", task="token-classification"
    )


def load_model(version="1"):
    """Load fine-tuned model checkpoint from mlflow artifacts.

    Returns: A dictionary of the saved individual components of
        either the Pipeline or the pre-trained model.
    """
    # TODO: want more ability to give load_model a model path?
    initialize_mlflow()
    return mlflow.transformers.load_model(
        f"models:/layoutlm_extractor/{version}", return_type="components"
    )


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


def draw_boxes_on_img(
    preds_or_labels,
    boxes,
    image,
    width,
    height,
    font=ImageFont.load_default(),
    unnormalize=False,
):
    """Draw bounding boxes on an image.

    Useful for visualizing result of inference.
    """
    draw = ImageDraw.Draw(image)
    label_color_lookup = {
        "subsidiary": "green",
        "loc": "red",
        "own_per": "orange",
    }
    for pred_or_label, box in zip(preds_or_labels, boxes):
        label = iob_to_label(pred_or_label).lower()
        if label == "other":
            continue
        if unnormalize:
            box = unnormalize_box(box, width, height)
        color = label_color_lookup[label]
        draw.rectangle(box, outline=color)
        draw.text((box[0] + 10, box[1] - 10), text=label, fill=color, font=font)
