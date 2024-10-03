"""Helper functions specific to Exhibit 21 model validation."""

from importlib import resources

import pandas as pd

from mozilla_sec_eia.models.sec10k.ex_21.inference import clean_extracted_df
from mozilla_sec_eia.models.sec10k.utils.cloud import get_metadata_filename


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


def remove_paragraph_tables_from_validation_data():
    """Remove paragraph layout docs from the Ex. 21 validation data."""
    labels_filename = "ex21_labels_with_paragraph.csv"
    labels_df = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data.validation_data")
        / labels_filename,
        comment="#",
    )
    hist_filename = "ex21_layout_histogram.csv"
    layout_hist_df = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data.validation_data") / hist_filename,
        comment="#",
    )
    paragraph_filenames = layout_hist_df[layout_hist_df["Layout Type"] == "Paragraph"][
        "Filename"
    ].unique()

    non_paragraph_df = labels_df[~labels_df["Filename"].isin(paragraph_filenames)]

    labels_filename_new = "ex21_labels.csv"
    non_paragraph_df.to_csv(
        resources.files("mozilla_sec_eia.package_data.validation_data")
        / labels_filename_new,
        index=False,
    )
