"""Util functions for training and predicting with LayoutLM on Ex. 21 tables."""

import mlflow
from dagster import InputContext, OutputContext
from PIL import ImageDraw, ImageFont
from transformers import (
    Trainer,
)

from mozilla_sec_eia.library.mlflow import MlflowBaseIOManager


def _load_pretrained_layoutlm(cache_path: str, version: str = "latest") -> dict:
    """Function to load layoutlm from mlflow."""
    path = f"models:/layoutlm_extractor/{version}"

    return mlflow.transformers.load_model(
        path, dst_path=cache_path, return_type="components"
    )


class LayoutlmIOManager(MlflowBaseIOManager):
    """Load and log models with mlflow tracking server."""

    version: int | None = None

    def handle_output(self, context: OutputContext, finetuned_model: Trainer):
        """Load metrics to mlflow run/experiment created by `MlflowInterface`."""
        model = {"model": finetuned_model.model, "tokenizer": finetuned_model.tokenizer}
        mlflow.transformers.log_model(
            model, artifact_path="layoutlm_extractor", task="token-classification"
        )

    def load_input(self, context: InputContext) -> dict:
        """Log metrics to mlflow run/experiment created by `MlflowInterface`."""
        cache_path = (
            self.mlflow_interface.dagster_home_path / "model_cache" / "layoutlm"
        )
        cache_path.mkdir(exist_ok=True, parents=True)
        return _load_pretrained_layoutlm(
            cache_path=cache_path,
            version=self.version,
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
