"""Implement functions for handling data from basic 10k filings (not exhibit 21)."""

import logging

import pandas as pd
from dagster import AssetIn, AssetOut, asset, multi_asset

from mozilla_sec_eia.library import validation_helpers

from .entities import basic_10k_extract_type, sec10k_extract_metadata_type
from .extract import (
    sec10k_filing_metadata,
    year_quarter_partitions,
)
from .utils.cloud import GCSArchive, Sec10K

logger = logging.getLogger(f"catalystcoop.{__name__}")


def _extract_10k(filing: Sec10K):
    """Extract basic company data from filing."""
    logger.info(f"Extracting 10K company data from filing: {filing.filename}")
    header = True
    current_block = None
    values = []
    filer_count = 0
    block_counts = {
        "company data": 0,
        "filing values": 0,
        "business address": 0,
        "mail address": 0,
        "former company": 0,
    }
    unmatched_keys = []
    for line in filing.filing_text.splitlines():
        match line.replace("\t", "").lower().split(":"):
            case ["filer", ""]:
                filer_count += 1
                header = False
            case [
                (
                    "company data"
                    | "filing values"
                    | "business address"
                    | "mail address"
                    | "former company"
                ) as block,
                "",
            ] if not header:
                current_block = block
                block_counts[current_block] += 1
            case [key, ""] if current_block is not None:
                key = f"{block}_{key}".replace(" ", "_")
                logger.warning(f"No value found for {key} for filing {filing.filename}")
                unmatched_keys.append(key)
            case [key, value] if current_block is not None:
                key = key.replace(" ", "_")
                values.append(
                    {
                        "filename": filing.filename,
                        "filer_count": filer_count - 1,
                        "block": current_block.replace(" ", "_"),
                        "block_count": block_counts[current_block] - 1,
                        "key": key.replace(" ", "_"),
                        "value": value,
                    }
                )
            case ["</sec-header>" | "</ims-header>"]:
                break
            case _ if header:
                continue

    return pd.DataFrame(values), filing.filename, unmatched_keys


def extract_filings(
    cloud_interface: GCSArchive,
    filings_to_extract: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract basic 10K data and return extracted data/metadata."""
    logger.info("Starting basic 10K extraction.")
    logger.info(f"Extracting {len(filings_to_extract)} filings.")

    extraction_metadata = pd.DataFrame(
        {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
    ).set_index("filename")
    extracted = pd.DataFrame()

    for filing in cloud_interface.iterate_filings(filings_to_extract):
        ext, filename, unmatched_keys = _extract_10k(filing)
        extraction_metadata.loc[filename, ["success", "unmatched_keys"]] = [
            len(ext) > 0,
            ",".join(unmatched_keys),
        ]
        extracted = pd.concat([extracted, ext])

    return (
        extraction_metadata,
        extracted.set_index(["filename", "filer_count", "block", "block_count", "key"]),
    )


@asset(dagster_type=basic_10k_extract_type)
def basic_10k_validation_set() -> pd.DataFrame:
    """Return dataframe containing basic 10k validation data."""
    return validation_helpers.load_validation_data(
        "basic_10k_labels.csv",
        index_cols=["filename", "filer_count", "block", "block_count", "key"],
    )


@asset(
    ins={
        "computed_df": AssetIn("basic_10k_company_info_validation"),
        "validation_df": AssetIn("basic_10k_validation_set"),
    },
    io_manager_key="mlflow_metrics_io_manager",
)
def basic_10k_extraction_validation_metrics(
    computed_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """Compute basic 10k extraction validation metrics."""
    return validation_helpers.pandas_compute_precision_recall(
        computed_df, validation_df, value_col="value"
    )


@asset
def basic_10k_validation_filing_metadata(
    cloud_interface: GCSArchive,
    basic_10k_validation_set: pd.DataFrame,
) -> pd.DataFrame:
    """Get sec 10k filing metadata from validation set."""
    filing_metadata = cloud_interface.get_metadata()
    return filing_metadata[
        filing_metadata["filename"].isin(
            basic_10k_validation_set.index.get_level_values("filename").unique()
        )
    ]


@multi_asset(
    outs={
        "basic_10k_extraction_metadata": AssetOut(
            io_manager_key="pandas_parquet_io_manager",
            dagster_type=sec10k_extract_metadata_type,
        ),
        "basic_10k_company_info": AssetOut(
            io_manager_key="pandas_parquet_io_manager",
            dagster_type=basic_10k_extract_type,
        ),
    },
    partitions_def=year_quarter_partitions,
)
def basic_10k_extract(
    cloud_interface: GCSArchive,
    sec10k_filing_metadata: pd.DataFrame,
):
    """Production asset for extracting basic 10k company info."""
    metadata, extracted = extract_filings(cloud_interface, sec10k_filing_metadata)
    return metadata, extracted


@multi_asset(
    outs={
        "basic_10k_extraction_metadata_validation": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager",
            dagster_type=sec10k_extract_metadata_type,
        ),
        "basic_10k_company_info_validation": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager",
            dagster_type=basic_10k_extract_type,
        ),
    },
)
def basic_10k_extract_validation(
    cloud_interface: GCSArchive,
    basic_10k_validation_filing_metadata: pd.DataFrame,
):
    """Production asset for extracting basic 10k company info."""
    metadata, extracted = extract_filings(
        cloud_interface, basic_10k_validation_filing_metadata
    )
    return metadata, extracted


production_assets = [basic_10k_extract, sec10k_filing_metadata]

validation_assets = [
    basic_10k_extract_validation,
    basic_10k_validation_set,
    basic_10k_validation_filing_metadata,
    basic_10k_extraction_validation_metrics,
]
