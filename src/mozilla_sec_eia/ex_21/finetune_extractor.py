"""Fine-tune LayoutLM to extract Ex. 21 tables.

This module uses a labeled training dataset to fine-tune
a LayoutLM model to extract unstructured Exhibit 21 tables
from SEC 10K filings.
"""

import mlflow
import numpy as np
from datasets import (
    Array2D,
    Array3D,
    Dataset,
    Features,
    Sequence,
    Value,
    load_metric,
)
from mozilla_sec_eia.ex_21.create_labeled_dataset import format_as_ner_annotations
from mozilla_sec_eia.utils.cloud import initialize_mlflow
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import default_data_collator


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


def _convert_ner_tags_to_id(ner_tags, label2id):
    return [int(label2id[ner_tag]) for ner_tag in ner_tags]


def prepare_dataset(annotations, processor, label2id):
    """Put the dataset in its final format for training LayoutLM."""
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


def train_model(model_output_dir="layoutlm_trainer", test_size=0.2):
    """Train LayoutLM model with labeled data.

    Arguments:
        model_output_dir: Path to directory where model
            checkpoints are saved.
    """
    initialize_mlflow()
    mlflow.set_experiment("/finetune-layoutlm")

    # Cache/prepare training data
    ner_annotations = format_as_ner_annotations()
    dataset = Dataset.from_list(ner_annotations)
    dataset = dataset.train_test_split(test_size=test_size)

    # Prepare model
    labels = ["O", "B-Subsidiary", "I-Subsidiary", "B-Loc", "I-Loc", "B-Own_Per"]
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in enumerate(labels)}
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
    )

    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )

    # Prepare our train & eval dataset
    column_names = dataset["train"].column_names
    features = Features(
        {
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(feature=Value(dtype="int64")),
        }
    )
    train_dataset = dataset["train"].map(
        lambda annotations: prepare_dataset(annotations, processor, label2id),
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    train_dataset.set_format("torch")
    eval_dataset = dataset["test"].map(
        lambda annotations: prepare_dataset(annotations, processor, label2id),
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    eval_dataset.set_format("torch")

    metric = load_metric("seqeval")
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        max_steps=1000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=lambda p: compute_metrics(p, metric=metric, label_list=labels),
    )

    with mlflow.start_run():
        trainer.train()

    # TODO: move into separate evaluation function?
    trainer.evaluate()
