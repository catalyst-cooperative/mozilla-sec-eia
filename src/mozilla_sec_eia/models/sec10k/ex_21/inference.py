"""Module for formatting inputs and performing inference with a fine-tuned LayoutLM model."""

import logging
import os
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dagster import ConfigurableResource
from datasets import Dataset
from transformers import (
    Pipeline,
    pipeline,
)
from transformers.tokenization_utils_base import BatchEncoding

from ..entities import Ex21CompanyOwnership, Sec10kExtractionMetadata
from ..utils.cloud import GCSArchive, get_metadata_filename
from ..utils.layoutlm import (
    get_id_label_conversions,
    iob_to_label,
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
from .train_extractor import BBOX_COLS, LABELS

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


def create_inference_dataset(
    filing_metadata: pd.DataFrame, cloud_interface: GCSArchive, has_labels: bool = False
) -> tuple[pd.DataFrame, Dataset]:
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
    for filename in image_dict:
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
    return extraction_metadata, dataset


def clean_extracted_df(extracted_df):
    """Perform basic cleaning on a dataframe extracted from an Ex. 21."""
    if extracted_df.empty:
        return extracted_df
    if "row" in extracted_df.columns:
        extracted_df = extracted_df.drop(columns=["row"])
    extracted_df["subsidiary"] = extracted_df["subsidiary"].str.strip().str.lower()
    # strip special chars from the start and end of the string
    extracted_df["subsidiary"] = extracted_df["subsidiary"].str.replace(
        r"^[^\w&\s]+|[^\w&\s]+$", "", regex=True
    )
    if "loc" in extracted_df.columns:
        extracted_df["loc"] = extracted_df["loc"].str.strip().str.lower()
        extracted_df["loc"] = extracted_df["loc"].str.replace(
            r"[^a-zA-Z&,\s]", "", regex=True
        )
    if "own_per" in extracted_df.columns:
        # remove special chars and letters
        extracted_df["own_per"] = extracted_df["own_per"].str.replace(
            r"[^\d.]", "", regex=True
        )
        # Find values with multiple decimal points
        extracted_df["own_per"] = extracted_df["own_per"].str.replace(
            r"(\d*\.\d+)\..*", r"\1", regex=True
        )
        extracted_df["own_per"] = extracted_df["own_per"].replace("", np.nan)
        extracted_df["own_per"] = extracted_df["own_per"].astype(
            "float64", errors="ignore"
        )
    # drop rows that have a null subsidiary value
    extracted_df = extracted_df.dropna(subset="subsidiary")
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


def _get_data(dataset):
    yield from dataset


class Exhibit21Extractor(ConfigurableResource):
    """Implement `Sec10kExtractor` interface for exhibit 21 data."""

    cloud_interface: GCSArchive
    name: str = "exhibit21_extractor"
    device: str = "cpu"
    has_labels: bool = False
    dataset_ind: list | None = None

    @contextmanager
    def setup_for_execution(self, context):
        """Set env variable to improve GPU memory access."""
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def extract_filings(
        self, dataset: Dataset, model, processor
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict entities with a fine-tuned model and extract Ex. 21 tables."""
        if self.dataset_ind:
            dataset = dataset.select(self.dataset_ind)

        # TODO: figure out device argument
        pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=processor,
            pipeline_class=LayoutLMInferencePipeline,
            device=self.device,
        )

        logits = []
        predictions = []
        hidden_states = []
        all_output_df = Ex21CompanyOwnership.example(size=0)
        extraction_metadata = Sec10kExtractionMetadata.example(size=0)
        for output_dict in pipe(_get_data(dataset)):
            # TODO: logits and predictions are useful for debugging, do something with them?
            logits.append(output_dict["logits"])
            predictions.append(output_dict["predictions"])
            # hidden_states.append(output_dict["final_hidden_state_embeddings"])
            output_df = output_dict["output_df"]
            if not output_df.empty:
                output_df = output_df.groupby("subsidiary").first().reset_index()
                filename = get_metadata_filename(output_df["id"].iloc[0])
                extraction_metadata.loc[filename, ["success"]] = True
            all_output_df = pd.concat([all_output_df, output_df])
        all_output_df.columns.name = None
        all_output_df = clean_extracted_df(all_output_df)
        all_output_df = all_output_df[["id", "subsidiary", "loc", "own_per"]]
        all_output_df = all_output_df.reset_index(drop=True)
        outputs_dict = {
            "all_output_df": all_output_df,
            "logits": logits,
            "predictions": predictions,
            # "final_hidden_state_embeddings": hidden_states,
        }
        return extraction_metadata, outputs_dict


def extract_filings(
    exhibit21_extractor: Exhibit21Extractor,
    filings: pd.DataFrame,
    layoutlm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create huggingface dataset from filings and perform extraction."""
    try:
        failed_metadata, dataset = create_inference_dataset(
            filing_metadata=filings,
            cloud_interface=exhibit21_extractor.cloud_interface,
            has_labels=exhibit21_extractor.has_labels,
        )
        metadata, outputs_dict = exhibit21_extractor.extract_filings(
            dataset,
            model=layoutlm["model"],
            processor=layoutlm["tokenizer"],
        )
        extracted = outputs_dict["all_output_df"]
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
            # TODO: if hidden states aren't needed, don't return in output
            outputs = self.model(**encoding, output_hidden_states=True)
            return {
                "logits": outputs.logits,
                "predictions": outputs.logits.argmax(-1).squeeze().tolist(),
                # TODO: debug this function
                # "final_hidden_state_embeddings": self._get_hidden_state_embeddings(
                # outputs=outputs, attention_mask=encoding["attention_mask"]
                # ),
                "raw_encoding": model_inputs["raw_encoding"],
                "doc_dict": model_inputs["doc_dict"],
            }

    def postprocess(self, output_dict):
        """Return logits, model predictions, and the extracted dataframe."""
        output_df = self._extract_table(output_dict)
        output_dict["output_df"] = output_df
        return output_dict

    def _get_hidden_state_embeddings(self, outputs, attention_mask):
        # This is a tuple with one tensor per model layer
        hidden_states = outputs.hidden_states
        # Final layer hidden state (batch_size, seq_length, hidden_size)
        final_hidden_state = hidden_states[-1]
        token_embeddings = final_hidden_state.squeeze(0)  # Remove batch dimension
        # just get embeddings for actual tokens
        # (ignore [CLS], [SEP] which are special tokens)
        attention_mask = attention_mask.squeeze(0)
        valid_token_embeddings = token_embeddings[attention_mask == 1]
        return valid_token_embeddings

    def _extract_table(self, output_dict):
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
        predictions = output_dict["predictions"]
        encoding = output_dict["raw_encoding"]
        doc_dict = output_dict["doc_dict"]

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
        # rows that are the first occurrence in a new group (subsidiary, loc, own_per)
        # should always have a B entity label. Manually override labels so this is true.
        first_in_group_df = df[
            (df["pred"].ne(df["pred"].shift())) & (df["pred"] != "other")
        ]
        first_in_group_df.loc[:, "iob_pred"] = (
            "B" + first_in_group_df["iob_pred"].str[1:]
        )
        df.update(first_in_group_df)
        # filter for just words that were labeled with non "other" entities
        # TODO: are the column top_left_y or top_left_y_pdf?
        entities_df = df.sort_values(by=["top_left_y", "top_left_x"])
        entities_df = entities_df[entities_df["pred"] != "other"]
        # boxes that have the same group label but are on different rows
        # should be updated to have two different B labels
        entities_df = separate_entities_by_row(entities_df)
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
        # separate_subsidiaries_by_bbox(grouped_df)
        output_df = grouped_df.pivot_table(
            index="row", columns="pred", values="word", aggfunc=lambda x: " ".join(x)
        ).reset_index()
        if output_df.empty:
            return output_df
        output_df.loc[:, "id"] = doc_dict["id"]
        return output_df


def separate_entities_by_row(df):
    # get the bounding boxes with labeled entities
    # get the average distance between boxes that don't share an x coordinate (with threshold)
    # if space between the y coordinates of two bboxes is greater than the average then
    # they can't have the same row
    # try using the fourth quartile
    # alternatively: run a document classifier, then if it's "subsidiary list" type
    # same subsidiaries can't share an x value
    # TODO: do we want to separate out by entity label?
    threshold = 0.2
    df["line_group"] = df["top_left_y"].transform(
        lambda y: (y // threshold).astype(int)
    )
    # Get the unique y-values for each line (group) per file
    line_positions = df.groupby(["line_group"])["top_left_y"].mean().reset_index()
    # Calculate the difference between adjacent y-values (i.e., distance between lines)
    line_positions["y_diff"] = line_positions["top_left_y"].diff()
    # Filter out NaN values and take the mean of the valid distances
    y_diffs = line_positions["y_diff"].dropna()
    avg_y_diff = round(y_diffs).quantile(0.3)
    # if an I labeled entity is more than avg_y_diff from it's previoius box then make it a B entity
    df["prev_y"] = df["top_left_y"].shift(1)
    df["prev_iob"] = df["iob_pred"].shift(1)

    # Apply vectorized condition:
    # 1. Current label is 'I'
    # 2. Previous row exists in the same file
    # 3. Y-distance exceeds the average y difference
    df["iob_pred"] = np.where(
        (df["iob_pred"].str[0] == "I")
        & ((df["top_left_y"] - df["prev_y"]) >= avg_y_diff),
        "B" + df["iob_pred"].str[1:],  # Update to 'B'
        df["iob_pred"],  # Keep as is
    )

    # Drop temporary columns
    df = df.drop(columns=["prev_y", "prev_iob"])

    return df


def id_paragraph_layout(page_df):
    """Identify the 'paragraph' type layout."""
    # if the average space between x and y bboxes is around the correct ratio
    # or try if it's below a certain threshold
    pass
