"""Fine-tune LayoutLM to extract Ex. 21 tables.

This module uses a labeled training dataset to fine-tune
a LayoutLM model to extract unstructured Exhibit 21 tables
from SEC 10K filings.
"""

from pathlib import Path

import numpy as np
from dagster import Config, asset
from datasets import (
    Array2D,
    Array3D,
    Dataset,
    Features,
    Sequence,
    Value,
    load_metric,
)
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import default_data_collator

from ..utils.layoutlm import get_id_label_conversions
from .create_labeled_dataset import format_as_ner_annotations

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


def compute_metrics(p, metric, label_list, return_entity_level_metrics=False):
    """Compute metrics to train and evaluate the model on."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def _prepare_dataset(annotations, processor, label2id):
    """Put the dataset in its final format for training LayoutLM."""

    def _convert_ner_tags_to_id(ner_tags, label2id):
        return [int(label2id[ner_tag]) for ner_tag in ner_tags]

    images = annotations["image"]
    words = annotations["tokens"]
    boxes = annotations["bboxes"]
    # Map over labels and convert to numeric id for each ner_tag
    ner_tags = [
        _convert_ner_tags_to_id(ner_tags, label2id)
        for ner_tags in annotations["ner_tags"]
    ]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        word_labels=ner_tags,
        truncation=True,
        padding="max_length",
    )

    return encoding


def load_test_train_set(
    processor: AutoProcessor, test_size: float, ner_annotations: list[dict]
):
    """Load training/test set and prepare for training or evaluation."""
    id2label, label2id = get_id_label_conversions(LABELS)
    # Cache/prepare training data
    dataset = Dataset.from_list(ner_annotations)

    # Prepare our train & eval dataset
    column_names = dataset.column_names
    features = Features(
        {
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(feature=Value(dtype="int64")),
        }
    )
    dataset = dataset.map(
        lambda annotations: _prepare_dataset(annotations, processor, label2id),
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    dataset.set_format("torch")
    split_dataset = dataset.train_test_split(test_size=test_size)
    return split_dataset["train"], split_dataset["test"]


class FineTuneConfig(Config):
    """Configuration to supply to `train_model`."""

    labeled_json_path: str = "sec10k_filings/labeled_jsons/"
    gcs_training_data_dir: str = "labeled"
    output_dir: str = "layoutlm_trainer"
    test_size: float = 0.2


@asset(io_manager_key="layoutlm_io_manager")
def layoutlm(
    config: FineTuneConfig,
):
    """Train LayoutLM model with labeled data."""
    # Prepare model
    id2label, label2id = get_id_label_conversions(LABELS)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
    )
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    ner_annotations = format_as_ner_annotations(
        labeled_json_path=Path(config.labeled_json_path),
        gcs_folder_name=config.gcs_training_data_dir,
    )
    # Get training/test data using pre-trained processor to prepare data
    train_dataset, eval_dataset = load_test_train_set(
        processor=processor, test_size=config.test_size, ner_annotations=ner_annotations
    )

    # Initialize our Trainer
    metric = load_metric("seqeval")
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=1000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_mps_device=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=lambda p: compute_metrics(p, metric=metric, label_list=LABELS),
    )

    # Train inside mlflow run. Mlflow will automatically handle logging training metrcis
    trainer.train()
    return trainer
