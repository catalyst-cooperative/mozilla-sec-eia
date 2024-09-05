"""Module for working with exhibit 21 data."""

import pandas as pd
from dagster import AssetIn, AssetOut, asset, multi_asset

from mozilla_sec_eia.library import validation_helpers
from mozilla_sec_eia.library.mlflow import mlflow_interface_resource

from ..extract import sec10k_extraction_asset_factory, sec10k_filing_metadata
from ..utils.cloud import GCSArchive, cloud_interface_resource, get_metadata_filename
from ..utils.layoutlm import LayoutlmResource
from .inference import Exhibit21Extractor, clean_extracted_df


@asset
def ex21_validation_set() -> pd.DataFrame:
    """Return dataframe containing basic 10k validation data."""
    return clean_ex21_validation_set(
        validation_helpers.load_validation_data("ex21_labels.csv")
    )


@asset
def ex21_validation_filing_metadata(
    cloud_interface: GCSArchive,
    ex21_validation_set: pd.DataFrame,
) -> pd.DataFrame:
    """Get sec 10k filing metadata from validation set."""
    filing_metadata = cloud_interface.get_metadata()
    return filing_metadata[
        filing_metadata["filename"].isin(ex21_validation_set["filename"].unique())
    ]


ex21_extracted_validation_asset_name = "ex21_validation"


@multi_asset(
    ins={
        "computed_df": AssetIn(ex21_extracted_validation_asset_name),
        "validation_df": AssetIn("ex21_validation_set"),
    },
    outs={
        "ex21_jaccard_per_table": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager"
        ),
        "ex21_precision_recall_per_table": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager"
        ),
        "ex21_incorrect_filenames": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager"
        ),
        "ex21_extraction_metrics": AssetOut(io_manager_key="mlflow_metrics_io_manager"),
    },
)
def ex21_validation_metrics(computed_df: pd.DataFrame, validation_df: pd.DataFrame):
    """Compute validation metrics for Ex. 21 extraction."""
    shared_cols = validation_df.columns.intersection(computed_df.columns)
    validation_df = validation_df.astype(computed_df[shared_cols].dtypes)
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
        if extracted_table_df.equals(validation_table_df):
            # TODO: strip llc and other company strings before comparison
            n_equal += 1
        else:
            incorrect_files.append(filename)
        # compute precision and recall for each column
        table_metrics_dict[filename] = {}
        jaccard_dict[filename] = {}
        for col in ["subsidiary", "loc", "own_per"]:
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


exhibit_21_extractor_resource = Exhibit21Extractor(
    cloud_interface=cloud_interface_resource,
    layoutlm=LayoutlmResource(mlflow_interface=mlflow_interface_resource),
)
ex21_production_extraction = sec10k_extraction_asset_factory(
    "ex21",
    exhibit_21_extractor_resource,
    extraction_metadata_asset_name="ex21_extraction_metadata",
    extracted_asset_name="ex21_company_ownership_info",
    io_manager_key="pandas_parquet_io_manager",
)


ex21_validation_extraction = sec10k_extraction_asset_factory(
    "ex21_validation",
    exhibit_21_extractor_resource,
    filing_metadata_asset_name="ex21_validation_filing_metadata",
    extraction_metadata_asset_name="ex21_extraction_validation_metadata",
    extracted_asset_name=ex21_extracted_validation_asset_name,
    partitions_def=None,
)

production_assets = [
    sec10k_filing_metadata,
    ex21_production_extraction,
]

validation_assets = [
    ex21_validation_set,
    ex21_validation_filing_metadata,
    ex21_validation_extraction,
    ex21_validation_metrics,
]
