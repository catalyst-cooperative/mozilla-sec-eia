"""Fine-tune LayoutLM to extract Ex. 21 tables.

This module uses a labeled training dataset to fine-tune
a LayoutLM model to extract unstructured Exhibit 21 tables
from SEC 10K filings.
"""

from pathlib import Path

import mlflow
import numpy as np
import torch
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
    Pipeline,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.data.data_collator import default_data_collator

from mozilla_sec_eia.ex_21.create_labeled_dataset import format_as_ner_annotations
from mozilla_sec_eia.utils.cloud import initialize_mlflow

LABELS = ["O", "B-Subsidiary", "I-Subsidiary", "B-Loc", "I-Loc", "B-Own_Per"]


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


def _get_id_label_conversions():
    """Return dicts mapping ids to labels and labels to ids."""
    id2label = dict(enumerate(LABELS))
    label2id = {v: k for k, v in enumerate(LABELS)}
    return id2label, label2id


def load_test_train_set(
    processor: AutoProcessor, test_size: float, ner_annotations: list[dict]
):
    """Load training/test set and prepare for training or evaluation."""
    id2label, label2id = _get_id_label_conversions()
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


def log_model(finetuned_model: Trainer):
    """Log fine-tuned model to mlflow artifacts."""
    model = {"model": finetuned_model.model, "tokenizer": finetuned_model.tokenizer}
    mlflow.transformers.log_model(
        model, artifact_path="layoutlm_extractor", task="token-classification"
    )


def load_model():
    """Load fine-tuned model from mlflow artifacts."""
    # TODO: want the ability to give load_model a model path?
    initialize_mlflow()
    return mlflow.transformers.load_model(
        "models:/layoutlm_extractor/1", return_type="components"
    )


def train_model(
    labeled_json_path: str,
    gcs_training_data_dir: str = "labeled",
    model_output_dir="layoutlm_trainer",
    test_size=0.2,
):
    """Train LayoutLM model with labeled data.

    Arguments:
        model_output_dir: Path to directory where model
            checkpoints are saved.
        test_size: Proportion of labeled dataset to use for test set.
    """
    # Prepare mlflow for tracking/logging model
    initialize_mlflow()
    mlflow.set_experiment("/finetune-layoutlmv3")

    # Prepare model
    id2label, label2id = _get_id_label_conversions()
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
    )
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    ner_annotations = format_as_ner_annotations(
        labeled_json_path=Path(labeled_json_path),
        gcs_folder_name=gcs_training_data_dir,
    )
    # Get training/test data using pre-trained processor to prepare data
    train_dataset, eval_dataset = load_test_train_set(
        processor=processor, test_size=test_size, ner_annotations=ner_annotations
    )

    # Initialize our Trainer
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
    with mlflow.start_run():
        trainer.train()
        log_model(trainer)


def inference(dataset, model, processor):
    """Predict entities with a fine-tuned model on Ex. 21 PDF.

    Returns:
        logits: The model outputs logits of shape (batch_size, seq_len, num_labels). Seq_len is
            the same as token length
        predictions: From the logits, we take the highest score for each token, using argmax.
            This serves as the predicted label for each token. It is shape (seq_len) or token
            length.

    """

    # TODO: make an extractor class with a train and predict method?
    def data(dataset):
        yield from dataset

    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=processor,
        pipeline_class=LayoutLMInferencePipeline,
    )

    logits = []
    predictions = []
    for logit, pred in pipe(data(dataset)):
        logits.append(logit)
        predictions.append(pred)
    return logits, predictions


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

    def preprocess(self, example):
        """Encode and tokenize model inputs."""
        image = example["image"]
        words = example["tokens"]
        boxes = example["bboxes"]
        # encoding = self.tokenizer(image, words, boxes=boxes, return_tensors="pt")
        encoding = self.tokenizer(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            # return_overflowing_tokens=True,
            max_length=512,  # this is the maximum max_length
            stride=128,
        )
        encoding["input_ids"] = encoding["input_ids"].to(torch.int64)
        encoding["attention_mask"] = encoding["attention_mask"].to(torch.int64)
        encoding["bbox"] = encoding["bbox"].to(torch.int64)
        return encoding

    def _forward(self, encoding):
        if torch.cuda.is_available():
            encoding.to("cuda")
            self.model.to("cuda")
        # since we're doing inference, we don't need gradient computation
        with torch.no_grad():
            output = self.model(**encoding)
            return output

    def postprocess(self, output):
        """Return logits and model predictions."""
        logits = output.logits
        predictions = output.logits.argmax(-1).squeeze().tolist()
        return logits, predictions
