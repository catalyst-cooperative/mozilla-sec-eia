"""Implement common utilities/functions for validating models."""

from importlib import resources

import pandas as pd
from dagster import AssetIn, AssetsDefinition, asset


def load_validation_data_asset_factory(
    asset_name: str,
    filename: str,
    index_cols: str | list[str] | None = None,
) -> AssetsDefinition:
    """Construct asset for loading validation data from CSV in `package_data`."""

    @asset(
        name=asset_name,
        io_manager_key="mlflow_pandas_artifact_io_manager",
    )
    def load_validation_data() -> pd.DataFrame:
        """Load csv with validation data from `package_data` directory."""
        df = pd.read_csv(
            resources.files("mozilla_sec_eia.package_data.validation_data") / filename
        )
        if index_cols is not None:
            df = df.set_index(index_cols)
        return df

    return load_validation_data


def pandas_precision_recall_asset_factory(
    validation_asset: str,
    computed_asset: str,
    value_col: str,
) -> AssetsDefinition:
    """Produce asset to compute precision and recall on pandas dataframe.

    The returned asset will take upstream computed/validation assets and compute
    precision/recall on `value_col`.

    Arg:
        validation_asset: Upstream asset containing dataframe of validation set.
        computed_asset: Upstream asset containing dataframe of computed data.
        value_col: Column to compare when computing metrics.
    """

    @asset(
        ins={
            "computed_set": AssetIn(computed_asset),
            "validation_set": AssetIn(validation_asset),
        },
        io_manager_key="mlflow_metrics_io_manager",
    )
    def pandas_compute_precision_recall(
        computed_set: pd.DataFrame,
        validation_set: pd.DataFrame,
    ) -> dict:
        """Asset which will return computed metrics from dataframes."""
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

    # Return new asset
    return pandas_compute_precision_recall
