"""Fine-tune LayoutLM to extract Ex. 21 tables.

This module uses a labeled training dataset to fine-tune
a LayoutLM model to extract unstructured Exhibit 21 tables
from SEC 10K filings.
"""

import numpy as np
from datasets import (
    Array2D,
    Array3D,
    ClassLabel,
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

LABELS = ["O", "B-Subsidiary", "I-Subsidiary", "B-Loc", "I-Loc", "B-Own_Per"]


class LayoutLMFineTuner:
    """A class to fine-tune LayooutLM for Ex. 21 extraction."""

    dataset: Dataset
    label_list: list
    id2label: dict
    label2id: dict
    class_label: ClassLabel
    column_names: list
    # Use the Auto API to load LayoutLMv3Processor based on the
    # checkpoint from the hub. Don't apply OCR because we
    # already got bboxes from the text PDF.
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    model: LayoutLMv3ForTokenClassification

    def __init__(self):
        """Initialize LayoutLMFineTuner."""
        self.set_train_test_dataset()
        self.metric = load_metric("seqeval")

    @classmethod
    def from_ner_annotations(cls, ner_annotations: list[dict], test_size=0.2):
        """Create LayoutLMFineTuner from labeled NER annotations."""
        dataset = Dataset.from_list(ner_annotations)
        dataset = dataset.train_test_split(test_size=test_size)
        label_list = LABELS
        id2label = dict(enumerate(label_list))
        label2id = {v: k for k, v in enumerate(label_list)}
        class_label = ClassLabel(names=label_list)
        column_names = dataset.column_names
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
        )
        return cls(
            dataset=dataset,
            label_list=label_list,
            id2label=id2label,
            label2id=label2id,
            class_label=class_label,
            column_names=column_names,
            model=model,
        )

    def _convert_ner_tags_to_id(self, ner_tags):
        return [int(self.label2id[ner_tag]) for ner_tag in ner_tags]

    def prepare_dataset(self, annotations):
        """Put the dataset in its final format for training LayoutLM."""
        images = annotations["image"]
        words = annotations["tokens"]
        boxes = annotations["bboxes"]
        # Map over labels and convert to numeric id for each ner_tag
        ner_tags = [
            self._convert_ner_tags_to_id(ner_tags)
            for ner_tags in annotations["ner_tags"]
        ]

        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            word_labels=ner_tags,
            truncation=True,
            padding="max_length",
        )

        return encoding

    def set_train_test_dataset(self):
        """Define a train and test set."""
        features = Features(
            {
                "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
                "input_ids": Sequence(feature=Value(dtype="int64")),
                "attention_mask": Sequence(Value(dtype="int64")),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "labels": Sequence(feature=Value(dtype="int64")),
            }
        )

        # Prepare our train & eval dataset
        self.train_dataset = self.dataset["train"].map(
            self.prepare_dataset,
            batched=True,
            remove_columns=self.column_names,
            features=features,
        )
        self.train_dataset.set_format("torch")
        self.test_dataset = self.dataset["test"].map(
            self.prepare_dataset,
            batched=True,
            remove_columns=self.column_names,
            features=features,
        )
        self.test_dataset.set_format("torch")

    def train_model(self, model_output_dir):
        """Train LayoutLM model with labeled data.

        Arguments:
            model_output_dir: Path to directory where model
                checkpoints are saved.
        """
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            max_steps=1000,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-5,
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.processor,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        # TODO: move into separate evaluation function?
        trainer.evaluate()

    def compute_metrics(self, p, return_entity_level_metrics=False):
        """Compute metrics to train and evaluate the model on."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [
                self.label_list[pred]
                for (pred, lab) in zip(prediction, label)
                if lab != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [
                self.label_list[lab]
                for (pred, lab) in zip(prediction, label)
                if lab != -100
            ]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
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
