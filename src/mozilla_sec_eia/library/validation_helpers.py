"""Implement common utilities/functions for validating models."""

import json
import re
from importlib import resources

import pandas as pd


def load_validation_data(
    filename: str, index_cols: list[str] | None = None
) -> pd.DataFrame:
    """Load csv with validation data from `package_data` directory."""
    df = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data.validation_data") / filename
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
    intersection = set(computed_df[value_col]).intersection(
        set(validation_df[value_col])
    )
    union = set(computed_df[value_col]).union(set(validation_df[value_col]))
    return float(len(intersection)) / float(len(union))


def strip_down_company_names(ser: pd.Series) -> pd.Series:
    """Strip LLC and other company name suffixes from a column.

    Used to compare subsidiary name columns during validation.
    """
    # this JSON is taken from PUDL package data (used for CompanyNameCleaner)
    json_source = (
        resources.files("mozilla_sec_eia.package_data") / "us_legal_forms.json"
    )
    with json_source.open() as json_file:
        legal_terms_dict = json.load(json_file)["legal_forms"]["en"]
    terms_list = list(legal_terms_dict.keys())
    for key in legal_terms_dict:
        terms_list += legal_terms_dict[key]
    reg = r"\b(?:" + "|".join(re.escape(word) for word in terms_list) + r")\b"
    ser = ser.replace(reg, "", regex=True)
    ser = ser.str.strip()
    # strip commas or other special chars that might be at the end of the name
    ser = ser.str.replace(r"[^[^\w&\s]+|[^\w&\s]+$]+", "", regex=True)
    ser = ser.str.strip()
    return ser


def fill_nulls_for_comparison(ser: pd.Series):
    """Fill nans in column to make similarity comparison more accurate.

    Used during validation of Ex. 21 table extraction.
    """
    ser = ser.fillna(999) if ser.dtype == float else ser.fillna("zzz")
    return ser
