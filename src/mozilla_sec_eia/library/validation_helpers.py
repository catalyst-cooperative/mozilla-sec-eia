"""Implement common utilities/functions for validating models."""

from importlib import resources

import pandas as pd


def load_validation_data(
    filename: str, index_cols: list[str] | None = None
) -> pd.DataFrame:
    """Load csv with validation data from `package_data` directory."""
    df = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data.validation_data") / filename,
        comment="#",
    )
    if index_cols is not None:
        df = df.set_index(index_cols)
    return df


def pandas_compute_precision_recall(
    computed_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    value_col: str,
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


def jaccard_similarity(
    computed_df: pd.DataFrame, validation_df: pd.DataFrame, value_col: str
) -> float:
    """Get the Jaccard similarity between two Series.

    Calculated as the intersection of the set divided
    by the union of the set.

    Args:
        computed_df: Extracted data.
        validation_df: Expected extraction results.
        value_col: Column to calculate Jaccard similarity on.
            Must be present in both dataframes.
    """
    # fill nans to make similarity comparison more accurate
    if (computed_df[value_col].dtype == float) and (
        validation_df[value_col].dtype == float
    ):
        computed_df[value_col] = computed_df[value_col].fillna(999)
        validation_df[value_col] = validation_df[value_col].fillna(999)
    else:
        computed_df[value_col] = computed_df[value_col].fillna("zzz")
        validation_df[value_col] = validation_df[value_col].fillna("zzz")
    intersection = set(computed_df[value_col]).intersection(
        set(validation_df[value_col])
    )
    union = set(computed_df[value_col]).union(set(validation_df[value_col]))
    return float(len(intersection)) / float(len(union))
