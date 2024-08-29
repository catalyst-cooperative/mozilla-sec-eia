"""Implement common utilities/functions for validating models."""

import pandas as pd
from dagster import OpDefinition, Out, op


def _pandas_compute_precision_recall(
    computed_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    value_col: str,
) -> dict:
    """Compute precision and recall for extraction compared to validation set.

    Arg:
        computed_set: Extracted data.
        validation_set: Expected extraction results.
        value_col: Column to compare when computing metrics.
    """
    # Get initial length of both sets
    computed_len = len(computed_set)
    validation_len = len(validation_set)

    # Get index of rows only in one set and make Null in other set
    idx_validation_only = validation_set.index.difference(computed_set.index)
    padded_compute_set = pd.concat(
        [
            computed_set[value_col],
            pd.Series([None] * len(idx_validation_only), index=idx_validation_only),
        ]
    ).sort_index()
    idx_compute_only = computed_set.index.difference(validation_set.index)
    padded_validation_set = pd.concat(
        [
            validation_set[value_col],
            pd.Series([None] * len(idx_compute_only), index=idx_compute_only),
        ]
    ).sort_index()

    true_positives = (padded_compute_set == padded_validation_set).sum()

    return {
        "precision": true_positives / computed_len,
        "recall": true_positives / validation_len,
    }


def pandas_precision_recall_op_factory(value_col: str) -> OpDefinition:
    """Return an op that will compute precision/recall on `value_col` of dataframe."""

    @op(
        out={
            "precision_recall_metrics": Out(io_manager_key="mlflow_metrics_io_manager")
        }
    )
    def _precision_recall_op(computed_set: pd.DataFrame, validation_set: pd.DataFrame):
        return _pandas_compute_precision_recall(computed_set, validation_set, value_col)

    return _precision_recall_op
