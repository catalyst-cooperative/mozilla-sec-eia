"""Module for formatting inputs and performing inference with a fine-tuned LayoutLM model."""

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from mozilla_sec_eia.ex_21.create_labeled_dataset import (
    BBOX_COLS_PDF,
    format_label_studio_output,
    get_image_dict,
)
from mozilla_sec_eia.ex_21.extractor import inference
from mozilla_sec_eia.utils.pdf import get_pdf_data_from_path, normalize_bboxes


def format_unlabeled_pdf_dataframe(pdfs_dir):
    """Read and format PDFs into a dataframe (without labels)."""
    inference_df = pd.DataFrame()
    for pdf_filename in os.listdir(pdfs_dir):
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


def create_inference_dataset(pdfs_dir, labeled_json_dir=None, has_labels=False):
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


def perform_inference(
    pdfs_dir: Path,
    model: LayoutLMv3ForTokenClassification,
    processor: AutoProcessor,
    dataset_ind: list = None,
    labeled_json_dir: Path = None,
    has_labels: bool = False,
):
    """Format PDFs for inference and run inference with a model.

    Arguments:
        pdfs_dir: Path to the directory with PDFs that are being used for inference.
        model: A fine-tuned LayoutLM model.
        processor: The tokenizer and encoder for model inputs.
        dataset_ind: A list of index numbers of dataset examples to be used for inference
            Default is None, in which the entire dataset created from the PDF directory
            is used.
        labeled_json_dir: Path to the directory with labeled JSONs from Label Studio. Cannot
            be None if has_labels is True.
        has_labels: Boolean, true if the data has associated labels that can be used in
            visualizing and validating results.
    """
    dataset = create_inference_dataset(
        pdfs_dir=pdfs_dir, labeled_json_dir=labeled_json_dir, has_labels=has_labels
    )
    if dataset_ind:
        dataset = dataset.select(dataset_ind)
    logit_list, prediction_list, output_dfs = inference(
        dataset=dataset, model=model, processor=processor
    )
    return logit_list, prediction_list, output_dfs
