"""Helper functions specific to Exhibit 21 model validation."""

import numpy as np
import pandas as pd

from mozilla_sec_eia.library import validation_helpers
from mozilla_sec_eia.models.sec10k.utils.cloud import get_metadata_filename


def ex21_validation_metrics(computed_df: pd.DataFrame, validation_df: pd.DataFrame):
    """Compute validation metrics for Ex. 21 extraction."""
    shared_cols = validation_df.columns.intersection(computed_df.columns)
    validation_df = validation_df.astype(computed_df[shared_cols].dtypes)
    # strip llc and other company name parts for the similarity comparison
    computed_df["subsidiary"] = validation_helpers.strip_down_company_names(
        computed_df["subsidiary"]
    )
    validation_df["subsidiary"] = validation_helpers.strip_down_company_names(
        validation_df["subsidiary"]
    )
    n_equal = 0
    validation_filenames = validation_df["id"].unique()
    n_files = len(validation_filenames)
    table_metrics_dict = {}
    jaccard_dict = {}
    incorrect_files = []
    # iterate through each file and check each extracted table
    for filename in validation_filenames:
        extracted_table_df = computed_df[computed_df["id"] == filename].reset_index(
            drop=True
        )
        validation_table_df = validation_df[
            validation_df["id"] == filename
        ].reset_index(drop=True)
        # check if the tables are exactly equal
        if extracted_table_df[["subsidiary", "loc", "own_per"]].equals(
            validation_table_df[["subsidiary", "loc", "own_per"]]
        ):
            n_equal += 1
        else:
            incorrect_files.append(filename)
        # compute jaccard sim + precision and recall for each column
        table_metrics_dict[filename] = {}
        jaccard_dict[filename] = {}
        for col in ["subsidiary", "loc", "own_per"]:
            extracted_table_df[col] = validation_helpers.fill_nulls_for_comparison(
                extracted_table_df[col]
            )
            validation_table_df[col] = validation_helpers.fill_nulls_for_comparison(
                validation_table_df[col]
            )
            table_prec_recall = validation_helpers.pandas_compute_precision_recall(
                extracted_table_df, validation_table_df, value_col=col
            )
            table_metrics_dict[filename][f"{col}_precision"] = table_prec_recall[
                "precision"
            ]
            table_metrics_dict[filename][f"{col}_recall"] = table_prec_recall["recall"]
            # get the jaccard similarity between columns
            jaccard_dict[filename][col] = validation_helpers.jaccard_similarity(
                computed_df=extracted_table_df,
                validation_df=validation_table_df,
                value_col=col,
            )

    jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient="index").reset_index()
    prec_recall_df = pd.DataFrame.from_dict(
        table_metrics_dict, orient="index"
    ).reset_index()

    return (
        jaccard_df,
        prec_recall_df,
        pd.DataFrame({"filename": incorrect_files}),
        {
            "table_accuracy": n_equal / n_files,
            "avg_subsidiary_jaccard_sim": jaccard_df["subsidiary"].sum() / n_files,
            "avg_location_jaccard_sim": jaccard_df["loc"].sum() / n_files,
            "avg_own_per_jaccard_sim": jaccard_df["own_per"].sum() / n_files,
            "avg_subsidiary_precision": prec_recall_df["subsidiary_precision"].sum()
            / n_files,
            "avg_location_precision": prec_recall_df["loc_precision"].sum() / n_files,
            "avg_own_per_precision": prec_recall_df["own_per_precision"].sum()
            / n_files,
            "avg_subsidiary_recall": prec_recall_df["subsidiary_recall"].sum()
            / n_files,
            "avg_location_recall": prec_recall_df["loc_recall"].sum() / n_files,
            "avg_own_per_recall": prec_recall_df["own_per_recall"].sum() / n_files,
        },
    )


def clean_extracted_df(extracted_df):
    """Perform basic cleaning on a dataframe extracted from an Ex. 21."""
    if extracted_df.empty:
        return extracted_df
    if "row" in extracted_df.columns:
        extracted_df = extracted_df.drop(columns=["row"])
    extracted_df["subsidiary"] = extracted_df["subsidiary"].str.strip().str.lower()
    # strip special chars from the start and end of the string
    extracted_df["subsidiary"] = extracted_df["subsidiary"].str.replace(
        r"^[^\w&\s]+|[^\w&\s]+$", "", regex=True
    )
    if "loc" in extracted_df.columns:
        extracted_df["loc"] = extracted_df["loc"].str.strip().str.lower()
        extracted_df["loc"] = extracted_df["loc"].str.replace(
            r"[^a-zA-Z&,\s]", "", regex=True
        )
    if "own_per" in extracted_df.columns:
        # remove special chars and letters
        extracted_df["own_per"] = extracted_df["own_per"].str.replace(
            r"[^\d.]", "", regex=True
        )
        # Find values with multiple decimal points
        extracted_df["own_per"] = extracted_df["own_per"].str.replace(
            r"(\d*\.\d+)\..*", r"\1", regex=True
        )
        extracted_df["own_per"] = extracted_df["own_per"].replace("", np.nan)
        extracted_df["own_per"] = extracted_df["own_per"].astype(
            "float64", errors="ignore"
        )
    # drop rows that have a null subsidiary value
    extracted_df = extracted_df.dropna(subset="subsidiary")
    return extracted_df


def clean_ex21_validation_set(validation_df: pd.DataFrame):
    """Clean Ex. 21 validation data to match extracted format."""
    validation_df = validation_df.rename(
        columns={
            "Filename": "id",
            "Subsidiary": "subsidiary",
            "Location of Incorporation": "loc",
            "Ownership Percentage": "own_per",
        }
    )
    validation_df["own_per"] = validation_df["own_per"].astype(str)
    validation_df["filename"] = validation_df["id"].apply(get_metadata_filename)
    validation_df = clean_extracted_df(validation_df)
    return validation_df
