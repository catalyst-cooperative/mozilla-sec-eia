"""Module for working with exhibit 21 data."""

import logging
import traceback

import pandas as pd
from dagster import (
    AssetOut,
    Out,
    graph_multi_asset,
    op,
)
from mlflow.pyfunc import PyFuncModel

from mozilla_sec_eia.library.mlflow import pyfunc_model_asset_factory

from ..entities import (
    Ex21CompanyOwnership,
    Ex21Layout,
    Sec10kExtractionMetadata,
    ex21_extract_type,
    ex21_layout_type,
    sec10k_extract_metadata_type,
)
from ..extract import chunk_filings, sec10k_filing_metadata, year_quarter_partitions
from ..utils.cloud import GCSArchive
from .data.inference import create_inference_dataset

logger = logging.getLogger(f"catalystcoop.{__name__}")


@op(
    out={
        "metadata": Out(dagster_type=sec10k_extract_metadata_type),
        "extracted": Out(dagster_type=ex21_extract_type),
    },
    tags={"model": "exhibit21_extractor"},
)
def extract_filing_chunk(
    parsed_chunk: tuple[pd.DataFrame, pd.DataFrame],
    exhibit21_extractor: PyFuncModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a set of filings and return results."""
    failed_parsing_metadata, inference_dataset = parsed_chunk
    extracted = Ex21CompanyOwnership.example(size=0)
    try:
        if not inference_dataset.empty:
            metadata, extracted = exhibit21_extractor.predict(inference_dataset)
            metadata = pd.concat([failed_parsing_metadata, metadata])
        else:
            metadata = failed_parsing_metadata
    except Exception as e:
        logger.warning(traceback.format_exc())
        logger.warning(f"Error while extracting filings: {inference_dataset['id']}")
        metadata = pd.DataFrame(
            {
                "filename": inference_dataset["id"],
                "success": [False] * len(inference_dataset),
                "notes": [str(e)] * len(inference_dataset),
            }
        ).set_index("filename")
    return metadata, extracted


@op(out={"layout": Out(dagster_type=ex21_layout_type)})
def classify_chunk_layouts(
    parsed_chunk: tuple[pd.DataFrame, pd.DataFrame],
    exhibit21_layout_classifier: PyFuncModel,
) -> pd.DataFrame:
    """Extract a set of filings and return results."""
    _, inference_dataset = parsed_chunk
    try:
        df = pd.DataFrame(
            {
                "filename": inference_dataset["id"],
                "paragraph": exhibit21_layout_classifier.predict(inference_dataset),
            }
        ).set_index("filename")
    except ValueError:
        df = pd.DataFrame(
            {
                "filename": inference_dataset["id"],
                "paragraph": [None] * len(inference_dataset),
            }
        ).set_index("filename")

    return df


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
        "layout": Out(
            io_manager_key="pandas_parquet_io_manager",
            dagster_type=ex21_layout_type,
        ),
    }
)
def collect_extracted_chunks(
    metadata_dfs: list[pd.DataFrame],
    extracted_dfs: list[pd.DataFrame],
    layout_dfs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect chunks of extracted filings."""
    metadata_dfs = [df for df in metadata_dfs if not df.empty]
    extracted_dfs = [df for df in extracted_dfs if not df.empty]
    metadata_df = pd.concat(metadata_dfs)
    extracted_df = pd.concat(extracted_dfs)
    layout_df = pd.concat(layout_dfs)
    return (
        Sec10kExtractionMetadata.validate(metadata_df),
        Ex21CompanyOwnership.validate(extracted_df),
        Ex21Layout.validate(layout_df),
    )


@op(
    out={
        "dataset": Out(
            io_manager_key="pickle_gcs_io_manager",
        ),
    }
)
def create_dataset(
    cloud_interface: GCSArchive, filings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct inference dataset from filing chunk."""
    return create_inference_dataset(
        filing_metadata=filings,
        cloud_interface=cloud_interface,
    )


exhibit21_extractor = pyfunc_model_asset_factory(
    name="exhibit21_extractor",
    mlflow_run_uri="runs:/426dd1b67cbd4677b6fa22b6b9d9173a/exhibit21_extractor",
)

exhibit21_layout_classifier = pyfunc_model_asset_factory(
    name="exhibit21_layout_classifier",
    mlflow_run_uri="runs:/cbdd906766b2427c93e9c957be6ea9c8/exhibit21_layout_classifier",
)


@graph_multi_asset(
    outs={
        "ex21_extraction_metadata": AssetOut(
            io_manager_key="pandas_parquet_io_manager"
        ),
        "ex21_company_ownership_info": AssetOut(
            io_manager_key="pandas_parquet_io_manager"
        ),
        "ex21_layout": AssetOut(io_manager_key="pandas_parquet_io_manager"),
    },
    partitions_def=year_quarter_partitions,
)
def ex21_extract(
    sec10k_filing_metadata: pd.DataFrame,
    exhibit21_extractor: PyFuncModel,
    exhibit21_layout_classifier: PyFuncModel,
):
    """Extract ownership info from exhibit 21 docs."""
    filing_chunks = chunk_filings(sec10k_filing_metadata)
    parsed_chunks = filing_chunks.map(create_dataset)
    layout_chunks = parsed_chunks.map(
        lambda chunk: classify_chunk_layouts(chunk, exhibit21_layout_classifier)
    )
    metadata_chunks, extracted_chunks = parsed_chunks.map(
        lambda chunk: extract_filing_chunk(chunk, exhibit21_extractor)
    )
    return collect_extracted_chunks(
        metadata_chunks.collect(), extracted_chunks.collect(), layout_chunks.collect()
    )


production_assets = [
    sec10k_filing_metadata,
    ex21_extract,
    exhibit21_extractor,
    exhibit21_layout_classifier,
]
