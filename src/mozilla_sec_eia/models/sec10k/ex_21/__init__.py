"""Module for working with exhibit 21 data."""

import logging

import pandas as pd
from dagster import (
    AssetOut,
    In,
    Out,
    graph_multi_asset,
    op,
)

from ..entities import (
    Ex21CompanyOwnership,
    Sec10kExtractionMetadata,
    ex21_extract_type,
    sec10k_extract_metadata_type,
)
from ..extract import chunk_filings, sec10k_filing_metadata, year_quarter_partitions
from .inference import extract_filings

logger = logging.getLogger(f"catalystcoop.{__name__}")


@op(
    out={
        "metadata": Out(dagster_type=sec10k_extract_metadata_type),
        "extracted": Out(dagster_type=ex21_extract_type),
    },
    ins={"exhibit21_extractor": In(input_manager_key="layoutlm_io_manager")},
)
def extract_filing_chunk(
    filings: pd.DataFrame,
    exhibit21_extractor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a set of filings and return results."""
    return extract_filings(filings, exhibit21_extractor)


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
    partitions_def=year_quarter_partitions,
)
def ex21_extract(
    sec10k_filing_metadata: pd.DataFrame,
):
    """Extract ownership info from exhibit 21 docs."""
    filing_chunks = chunk_filings(sec10k_filing_metadata)
    metadata_chunks, extracted_chunks = filing_chunks.map(
        lambda filings: extract_filing_chunk(filings)
    )
    metadata, extracted = collect_extracted_chunks(
        metadata_chunks.collect(), extracted_chunks.collect()
    )

    return metadata, extracted


production_assets = [sec10k_filing_metadata, ex21_extract]
