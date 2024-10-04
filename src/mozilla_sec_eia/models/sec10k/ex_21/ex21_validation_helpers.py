"""Helper functions specific to Exhibit 21 model validation."""

import numpy as np
import pandas as pd

from mozilla_sec_eia.models.sec10k.utils.cloud import get_metadata_filename


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
