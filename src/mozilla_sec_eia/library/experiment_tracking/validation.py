"""Implement common utilities/functions for validating models."""

from importlib import resources

import pandas as pd
from dagster import Config, Out, op


class LoadValidationConfig(Config):
    """Configuration for loading validation data."""

    filename: str


@op(
    required_resource_keys=["experiment_tracker"],
    out={"validation_set": Out(io_manager_key="mlflow_pandas_artifact_io_manager")},
)
def load_validation_data(config: LoadValidationConfig) -> pd.DataFrame:
    """Load csv with validation data from `package_data` directory."""
    return pd.read_csv(
        resources.files("mozilla_sec_eia.package_data.validation_data")
        / config.filename
    )


class PandasPrecisionRecallConfig(Config):
    """Configuration for computing precision/recall from pandas dataframe."""

    value_col: str


@op(
    out={
        "pandas_precision_recall_metrics": Out(
            io_manager_key="mlflow_metrics_io_manager"
        )
    }
)
def pandas_compute_precision_recall(
    config: PandasPrecisionRecallConfig,
    computed_set: pd.DataFrame,
    validation_set: pd.DataFrame,
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
            computed_set[config.value_col],
            pd.Series([None] * len(idx_validation_only), index=idx_validation_only),
        ]
    ).sort_index()
    idx_compute_only = computed_set.index.difference(validation_set.index)
    padded_validation_set = pd.concat(
        [
            validation_set[config.value_col],
            pd.Series([None] * len(idx_compute_only), index=idx_compute_only),
        ]
    ).sort_index()

    true_positives = (padded_compute_set == padded_validation_set).sum()

    return {
        "precision": true_positives / computed_len,
        "recall": true_positives / validation_len,
    }
