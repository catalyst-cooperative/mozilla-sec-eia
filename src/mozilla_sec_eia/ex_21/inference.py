"""Module for formatting inputs and performing inference with a fine-tuned LayoutLM model."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Pipeline,
    pipeline,
)
from transformers.tokenization_utils_base import BatchEncoding

from mozilla_sec_eia.ex_21.create_labeled_dataset import (
    BBOX_COLS_PDF,
    format_label_studio_output,
    get_image_dict,
)
from mozilla_sec_eia.ex_21.train_extractor import BBOX_COLS, LABELS
from mozilla_sec_eia.utils.cloud import get_metadata_filename
from mozilla_sec_eia.utils.layoutlm import (
    get_id_label_conversions,
    iob_to_label,
    normalize_bboxes,
)
from mozilla_sec_eia.utils.pdf import (
    get_pdf_data_from_path,
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


def create_inference_dataset(pdfs_dir: Path, labeled_json_dir=None, has_labels=False):
    """Create a Hugging Face Dataset from PDFs for inference."""
    if has_labels:
        inference_df = format_label_studio_output(
            labeled_json_dir=labeled_json_dir, pdfs_dir=pdfs_dir
        )
    else:
        inference_df = format_unlabeled_pdf_dataframe(pdfs_dir=pdfs_dir)
    image_dict = get_image_dict(pdfs_dir)
    doc_filenames = inference_df["id"].unique()
    annotations = []
    for filename in doc_filenames:
        annotation = {
            "id": filename,
            "tokens": inference_df.groupby("id")["text"].apply(list).loc[filename],
            "bboxes": inference_df.loc[inference_df["id"] == filename, :][BBOX_COLS_PDF]
            .to_numpy()
            .tolist(),
            "image": image_dict[filename],
        }
        if has_labels:
            annotation["ner_tags"] = (
                inference_df.groupby("id")["ner_tag"].apply(list).loc[filename]
            )
        annotations.append(annotation)

    dataset = Dataset.from_list(annotations)
    return dataset


def clean_extracted_df(extracted_df):
    """Perform basic cleaning on a dataframe extracted from an Ex. 21."""
    if extracted_df.empty:
        return extracted_df
    if "row" in extracted_df.columns:
        extracted_df = extracted_df.drop(columns=["row"])
    extracted_df["subsidiary"] = extracted_df["subsidiary"].str.strip().str.lower()
    if "loc" in extracted_df.columns:
        extracted_df["loc"] = extracted_df["loc"].str.strip().str.lower()
    if "own_per" in extracted_df.columns:
        # remove special chars and letters
        extracted_df["own_per"] = extracted_df["own_per"].str.replace(
            r"[^\d.]", "", regex=True
        )
        extracted_df["own_per"] = extracted_df["own_per"].replace("", np.nan)
        extracted_df["own_per"] = extracted_df["own_per"].astype(
            "float64", errors="ignore"
        )
    return extracted_df


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


def _get_data(dataset):
    yield from dataset


def perform_inference(
    pdfs_dir: Path,
    model: LayoutLMv3ForTokenClassification,
    processor: AutoProcessor,
    extraction_metadata: pd.DataFrame,
    dataset_ind: list = None,
    labeled_json_dir: Path = None,
    has_labels: bool = False,
    device="cpu",
):
    """Predict entities with a fine-tuned model and extract Ex. 21 tables.

    This function starts by creating a HuggingFace dataset from PDFs in `pdfs_dir`
    that the model can then perform inference on (`create_inference_dataset`).
    Then it creates an instance of the custom LayoutLM inference pipeline and
    runs the dataset through the pipeline. The pipeline outputs logits, predictions,
    and an output dataframe with extracted Ex. 21 table.

    Arguments:
        pdfs_dir: Path to the directory with PDFs that are being used for inference.
        model: A fine-tuned LayoutLM model.
        processor: The tokenizer and encoder for model inputs.
        extraction_metadata: A dataframe to track extraction success metrics. Should
            have columns 'filename' and 'success'.
        dataset_ind: A list of index numbers of dataset records to be used for inference
            Default is None, in which the entire dataset created from the PDF directory
            is used.
        labeled_json_dir: Path to the directory with labeled JSONs from Label Studio. Cannot
            be None if has_labels is True.
        has_labels: Boolean, true if the data has associated labels that can be used in
            visualizing and validating results.
        device: String or int, specify what computation device to use for inference
            i.e. "mps", "cpu", "cuda"

    Returns:
        logits: A list of logits. The list is the length of the number of documents in the
            dataset (number of PDFs in pdfs_dir). Each logit object in the list is of
            shape (batch_size, seq_len, num_labels). Seq_len is
            the same as token length (512 in this case).
        predictions: A list of predictions. The list is the length of the number of documents
            in the dataset (number of PDFs in pdfs_dir).
            From the logits, we take the highest score for each token, using argmax.
            This serves as the predicted label for each token. It is shape (seq_len) or token
            length.
        output_dfs: The extracted Ex. 21 tables. This is one big dataframe with an ID column
            that is the filename of the extracted Ex. 21. Dataframe contains columns id,
            subsidiary, loc, own_per.
    """
    dataset = create_inference_dataset(
        pdfs_dir=pdfs_dir, labeled_json_dir=labeled_json_dir, has_labels=has_labels
    )
    if dataset_ind:
        dataset = dataset.select(dataset_ind)

    # TODO: figure out device argument
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=processor,
        pipeline_class=LayoutLMInferencePipeline,
        device=device,
    )

    logits = []
    predictions = []
    all_output_df = pd.DataFrame(columns=["id", "subsidiary", "loc", "own_per"])
    for logit, pred, output_df in pipe(_get_data(dataset)):
        logits.append(logit)
        predictions.append(pred)
        if not output_df.empty:
            filename = get_metadata_filename(output_df["id"].iloc[0])
            extraction_metadata.loc[filename, ["success"]] = True
        all_output_df = pd.concat([all_output_df, output_df])
    all_output_df.columns.name = None
    all_output_df = clean_extracted_df(all_output_df)
    all_output_df = all_output_df[["id", "subsidiary", "loc", "own_per"]]
    all_output_df = all_output_df.reset_index(drop=True)
    return logits, predictions, all_output_df, extraction_metadata


class LayoutLMInferencePipeline(Pipeline):
    """Pipeline for performing inference with fine-tuned LayoutLM."""

    def __init__(self, *args, **kwargs):
        """Initialize LayoutLMInferencePipeline."""
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, doc_dict):
        """Encode and tokenize model inputs."""
        image = doc_dict["image"]
        words = doc_dict["tokens"]
        boxes = doc_dict["bboxes"]
        encoding = self.tokenizer(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,  # this is the maximum max_length
            stride=128,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )
        model_inputs = {}
        model_inputs["raw_encoding"] = encoding.copy()
        model_inputs["doc_dict"] = doc_dict
        model_inputs["offset_mapping"] = encoding.pop("offset_mapping")
        model_inputs["sample_mapping"] = encoding.pop("overflow_to_sample_mapping")
        # TODO: do we actually need to make these into ints?
        encoding["input_ids"] = encoding["input_ids"].to(torch.int64)
        encoding["attention_mask"] = encoding["attention_mask"].to(torch.int64)
        encoding["bbox"] = encoding["bbox"].to(torch.int64)
        encoding["pixel_values"] = torch.stack(encoding["pixel_values"])
        model_inputs["encoding"] = encoding
        return model_inputs

    def _forward(self, model_inputs):
        # encoding is passed as a UserDict in the model_inputs dictionary
        # turn it back into a BatchEncoding
        encoding = BatchEncoding(model_inputs["encoding"])
        if torch.cuda.is_available():
            encoding.to("cuda")
            self.model.to("cuda")
        # since we're doing inference, we don't need gradient computation
        with torch.no_grad():
            output = self.model(**encoding)
            return {
                "logits": output.logits,
                "predictions": output.logits.argmax(-1).squeeze().tolist(),
                "raw_encoding": model_inputs["raw_encoding"],
                "doc_dict": model_inputs["doc_dict"],
            }

    def postprocess(self, all_outputs):
        """Return logits, model predictions, and the extracted dataframe."""
        logits = all_outputs["logits"]
        predictions = all_outputs["logits"].argmax(-1).squeeze().tolist()
        output_df = self.extract_table(all_outputs)
        return logits, predictions, output_df

    def extract_table(self, all_outputs):
        """Extract a structured table from a set of inference predictions.

        This function essentially works by stacking bounding boxes and predictions
        into a dataframe and going from left to right and top to bottom. Then, every
        every time a new subsidiary entity is encountered, it assigns a new group or
        "row" to that subsidiary. Next, location and ownership percentage words/labeled
        entities in between these subsidiary groups are assigned to a subsidiary row/group.
        Finally, this is all formatted into a dataframe with an ID column from the original
        filename and a basic cleaning function normalizes strings.
        """
        # TODO: when model more mature, break this into sub functions to make it
        # clearer what's going on
        predictions = all_outputs["predictions"]
        encoding = all_outputs["raw_encoding"]
        doc_dict = all_outputs["doc_dict"]

        token_boxes_tensor = encoding["bbox"].flatten(start_dim=0, end_dim=1)
        predictions_tensor = torch.tensor(predictions)
        mode_predictions = get_flattened_mode_predictions(
            token_boxes_tensor, predictions_tensor
        )
        token_boxes = encoding["bbox"].flatten(start_dim=0, end_dim=1).tolist()
        predicted_labels = [
            self.model.config.id2label[pred] for pred in mode_predictions
        ]
        simple_preds = [iob_to_label(pred).lower() for pred in predicted_labels]

        df = pd.DataFrame(data=token_boxes, columns=BBOX_COLS)
        df.loc[:, "iob_pred"] = predicted_labels
        df.loc[:, "pred"] = simple_preds
        invalid_mask = (
            (df["top_left_x"] == 0)
            & (df["top_left_y"] == 0)
            & (df["bottom_right_x"] == 0)
            & (df["bottom_right_y"] == 0)
        )
        df = df[~invalid_mask]
        # we want to get actual words on the dataframe, not just subwords that correspond to tokens
        # subwords from the same word share the same bounding box coordinates
        # so we merge the original words onto our dataframe on bbox coordinates
        words_df = pd.DataFrame(data=doc_dict["bboxes"], columns=BBOX_COLS)
        words_df.loc[:, "word"] = doc_dict["tokens"]
        df = df.merge(words_df, how="left", on=BBOX_COLS).drop_duplicates(
            subset=BBOX_COLS + ["pred", "word"]
        )
        # filter for just words that were labeled with non "other" entities
        entities_df = df.sort_values(by=["top_left_y", "top_left_x"])
        entities_df = entities_df[entities_df["pred"] != "other"]
        # words are labeled with IOB format which stands for inside, outside, beginning
        # merge B and I entities to form one entity group
        # (i.e. "B-Subsidiary" and "I-Subsidiary" become just "subsidiary"), assign a group ID
        entities_df["group"] = (entities_df["iob_pred"].str.startswith("B-")).cumsum()
        grouped_df = (
            entities_df.groupby(["group", "pred"])["word"]
            .apply(" ".join)
            .reset_index()[["pred", "word"]]
        )
        # assign a new row every time there's a new subsidiary
        grouped_df["row"] = (grouped_df["pred"].str.startswith("subsidiary")).cumsum()
        output_df = grouped_df.pivot_table(
            index="row", columns="pred", values="word", aggfunc=lambda x: " ".join(x)
        ).reset_index()
        if output_df.empty:
            return output_df
        output_df.loc[:, "id"] = doc_dict["id"]
        return output_df
