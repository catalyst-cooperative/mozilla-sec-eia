"""Module for working with exhibit 21 data."""

import logging

import mlflow
import pandas as pd
from dagster import AssetIn, AssetOut, Out, asset, graph_multi_asset, multi_asset, op

from mozilla_sec_eia.library import validation_helpers
from mozilla_sec_eia.library.mlflow import MlflowInterface, mlflow_interface_resource
from mozilla_sec_eia.models.sec10k.ex_21.ex21_validation_helpers import (
    clean_ex21_validation_set,
)

from ..entities import (
    Ex21CompanyOwnership,
    Sec10kExtractionMetadata,
    ex21_extract_type,
    sec10k_extract_metadata_type,
)
from ..extract import chunk_filings, sec10k_filing_metadata, year_quarter_partitions
from ..utils.cloud import GCSArchive, cloud_interface_resource, get_metadata_filename
from .inference import Exhibit21Extractor, clean_extracted_df, extract_filings

logger = logging.getLogger(f"catalystcoop.{__name__}")


@asset(dagster_type=ex21_extract_type)
def ex21_validation_set() -> pd.DataFrame:
    """Return dataframe containing exhibit 21 validation data."""
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
        filing_metadata.index.isin(ex21_validation_set["filename"].unique())
    ]


@multi_asset(
    ins={
        "computed_df": AssetIn("ex21_company_ownership_info_validation"),
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


@op(
    out={
        "metadata": Out(dagster_type=sec10k_extract_metadata_type),
        "extracted": Out(dagster_type=ex21_extract_type),
    }
)
def extract_filing_chunk(
    exhibit21_extractor: Exhibit21Extractor,
    filings: pd.DataFrame,
    layoutlm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a set of filings and return results."""
    return extract_filings(exhibit21_extractor, filings, layoutlm)


@op(
    out={
        "metadata": Out(
            io_manager_key="pandas_parquet_io_manager",
            dagster_type=sec10k_extract_metadata_type,
        ),
        "extracted": Out(
            io_manager_key="pandas_parquet_io_manager",
            dagster_type=ex21_extract_type,
        ),
    }
)
def collect_extracted_chunks(
    metadata_dfs: list[pd.DataFrame],
    extracted_dfs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect chunks of extracted filings."""
    metadata_dfs = [df for df in metadata_dfs if not df.empty]
    extracted_dfs = [df for df in extracted_dfs if not df.empty]
    metadata_df = pd.concat(metadata_dfs)
    extracted_df = pd.concat(extracted_dfs)
    return (
        Sec10kExtractionMetadata.validate(metadata_df),
        Ex21CompanyOwnership.validate(extracted_df),
    )


@graph_multi_asset(
    outs={
        "ex21_extraction_metadata": AssetOut(
            io_manager_key="pandas_parquet_io_manager"
        ),
        "ex21_company_ownership_info": AssetOut(
            io_manager_key="pandas_parquet_io_manager"
        ),
    },
    ins={"layoutlm": AssetIn(input_manager_key="layoutlm_io_manager")},
    partitions_def=year_quarter_partitions,
)
def ex21_extract(
    sec10k_filing_metadata: pd.DataFrame,
    layoutlm,
):
    """Extract ownership info from exhibit 21 docs."""
    filing_chunks = chunk_filings(sec10k_filing_metadata)
    metadata_chunks, extracted_chunks = filing_chunks.map(
        lambda filings: extract_filing_chunk(filings, layoutlm)
    )
    metadata, extracted = collect_extracted_chunks(
        metadata_chunks.collect(), extracted_chunks.collect()
    )

    return metadata, extracted


@multi_asset(
    outs={
        "ex21_extraction_metadata_validation": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager",
            dagster_type=sec10k_extract_metadata_type,
        ),
        "ex21_company_ownership_info_validation": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager",
            dagster_type=ex21_extract_type,
        ),
    },
    ins={"layoutlm": AssetIn(input_manager_key="layoutlm_io_manager")},
)
def ex21_extract_validation(
    ex21_validation_filing_metadata: pd.DataFrame,
    exhibit21_extractor: Exhibit21Extractor,
    layoutlm,
):
    """Extract ownership info from exhibit 21 docs."""
    return extract_filings(
        exhibit21_extractor, ex21_validation_filing_metadata, layoutlm
    )


exhibit_21_extractor_resource = Exhibit21Extractor(
    cloud_interface=cloud_interface_resource,
)

production_assets = [sec10k_filing_metadata, ex21_extract]

validation_assets = [
    ex21_validation_set,
    ex21_validation_filing_metadata,
    ex21_extract_validation,
    ex21_validation_metrics,
]
