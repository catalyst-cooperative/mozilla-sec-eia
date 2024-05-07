"""Fine-tune LayoutLM to extract Ex. 21 tables.

This module uses a labeled training dataset to fine-tune
a LayoutLM model to extract unstructured Exhibit 21 tables
from SEC 10K filings.
"""

from datasets import (
    Dataset,
    ClassLabel,
    Features,
    Sequence,
    Value,
    Array2D,
    Array3D,
    load_metric,
)
import torch
import numpy as np
from transformers import AutoProcessor


class LayoutLMFineTuner:
    """A class to fine-tune LayooutLM for Ex. 21 extraction."""

    ner_annotations: dict
    dataset: Dataset
    label_list: list
    id2label: dict
    label2id: dict
    class_label: ClassLabel
    column_names: list

    # TODO: have a from_ner_annotations class method with
    # dataset = Dataset.from_list(ner_annotations)

    def _convert_ner_tags_to_id(self, ner_tags):
        return [int(self.label2id[ner_tag]) for ner_tag in ner_tags]

    # This function is used to put the Dataset in its final format for training LayoutLM
    def prepare_dataset(self, processor):
        """Put the dataset in its final format for training LayoutLM"""
        images = self.ner_annotations["image"]
        words = self.ner_annotations["tokens"]
        boxes = self.ner_annotations["bboxes"]
        # Map over labels and convert to numeric id for each ner_tag
        ner_tags = [
            self._convert_ner_tags_to_id(ner_tags)
            for ner_tags in self.ner_annotations["ner_tags"]
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

    def compute_metrics(self, p, metric, return_entity_level_metrics=False):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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
