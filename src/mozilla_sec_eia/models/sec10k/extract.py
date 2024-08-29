"""Implement top level extraction methods and tooling."""

import logging
import math

import mlflow
import numpy as np
import pandas as pd
import pandera as pa
from dagster import (
    Config,
    DynamicOut,
    DynamicOutput,
    GraphDefinition,
    GraphOut,
    In,
    OpDefinition,
    Out,
    Output,
    graph,
    op,
)

from mozilla_sec_eia.library.models import pudl_model

from . import basic_10k
from .utils.cloud import GCSArchive, cloud_interface_resource

logger = logging.getLogger(f"catalystcoop.{__name__}")

DATASETS = ["ex21", "basic_10k"]


class ExtractionMetadataSchema(pa.DataFrameModel):
    """Define the required schema for extraction metadata.

    Extra columns are permitted, but these are required for computing extraction metrics.
    """

    filename: pa.typing.Index[str] = pa.Field(check_name=True)
    success: bool = pa.Field(coerce=True)


@op
def get_filing_metadata(
    cloud_interface: GCSArchive,
) -> pd.DataFrame:
    """Return filing metadata."""
    return cloud_interface.get_metadata()


class ChunkFilingsConfig(Config):
    """Config how many filings are extracted and chunk_size for extraction."""

    chunk_size: int = 1000


@op(out=DynamicOut())
def chunk_filings(
    config: ChunkFilingsConfig,
    filings_to_extract: pd.DataFrame,
) -> pd.DataFrame:
    """Split filings into chunks for parallel extraction."""
    for i, chunk in enumerate(
        np.array_split(
            filings_to_extract, math.ceil(len(filings_to_extract) / config.chunk_size)
        )
    ):
        yield DynamicOutput(chunk, mapping_key=str(i))


class GetMostRecentRunResultsConfig(Config):
    """Configuration specifying whether to get run results and continue."""

    continue_run: bool = False


@op(required_resource_keys=["experiment_tracker"])
def log_extraction_data(
    metadata: pd.DataFrame,
    extraction_metadata: pd.DataFrame,
    extracted: pd.DataFrame,
):
    """Log results from extraction run."""
    mlflow.log_metrics(
        {
            "num_failed": (~extraction_metadata["success"]).sum(),
            "ratio_extracted": len(extraction_metadata) / len(metadata),
        }
    )

    return extraction_metadata, extracted


@op(
    required_resource_keys=["experiment_tracker"],
    out={
        "extraction_metadata": Out(io_manager_key="mlflow_pandas_artifact_io_manager"),
        "extracted": Out(io_manager_key="mlflow_pandas_artifact_io_manager"),
    },
)
def merge_extracted_data(
    extraction_metadata: list[pd.DataFrame],
    extracted: list[pd.DataFrame],
    previous_run_extraction_metadata: pd.DataFrame,
    previous_run_extracted_data: pd.DataFrame,
):
    """Data is extracted in parallel ops, merge these plus any data from previous run."""
    extraction_metadata = pd.concat(
        extraction_metadata + [previous_run_extraction_metadata]
    )
    extracted = pd.concat(extracted + [previous_run_extracted_data])
    # Use metadata to log generic metrics
    extraction_metadata = ExtractionMetadataSchema.validate(extraction_metadata)

    return extraction_metadata, extracted


class FilingsToExtractConfig(Config):
    """Define configuration for filtering filings to extract."""

    num_filings: int = -1


@op
def get_filings_to_extract(
    config: FilingsToExtractConfig,
    filing_metadata: pd.DataFrame,
    previous_extraction_metadata: pd.DataFrame,
    previous_extracted: pd.DataFrame,
):
    """Filter out any previously extracted filings and sub-sample to `num_filings`."""
    filings_to_extract = filing_metadata
    if config.num_filings > 0:
        filings_to_extract = filings_to_extract.sample(config.num_filings)

    filings_to_extract = filings_to_extract[
        ~filings_to_extract["filename"].isin(previous_extraction_metadata.index)
    ]
    return filings_to_extract


def extract_graph_factory(
    dataset_name: str,
    extract_op: OpDefinition | GraphDefinition,
):
    """Produce a `pudl_model` to extract data from sec10k filings."""
    experiment_name = f"{dataset_name}_extraction"

    @graph(name=experiment_name)
    def extract_filings(previous_extraction_metadata, previous_extracted):
        metadata = get_filing_metadata()
        filings_to_extract = get_filings_to_extract(
            metadata,
            previous_extraction_metadata,
            previous_extracted,
        )

        filing_chunks = chunk_filings(filings_to_extract)
        extraction_metadata, extracted = filing_chunks.map(extract_op)
        extraction_metadata, extracted = merge_extracted_data(
            extraction_metadata.collect(),
            extracted.collect(),
            previous_extraction_metadata,
            previous_extracted,
        )

        return log_extraction_data(
            metadata,
            extraction_metadata,
            extracted,
        )

    return extract_filings


@op(
    ins={
        "extraction_metadata": In(
            input_manager_key="previous_run_mlflow_pandas_artifact_io_manager"
        ),
        "extracted": In(
            input_manager_key="previous_run_mlflow_pandas_artifact_io_manager"
        ),
    }
)
def get_previous_run_data(
    continue_previous_run, extraction_metadata: pd.DataFrame, extracted: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return previous run data loaded by io-manager."""
    extraction_metadata = ExtractionMetadataSchema.validate(extraction_metadata)

    return extraction_metadata, extracted


@op
def get_empty_run_data(start_new_run):
    """Return empty dataframes representing run metadata and extracted data."""
    extraction_metadata = pd.DataFrame(
        {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
    ).set_index("filename")
    extracted = pd.DataFrame()

    return extraction_metadata, extracted


class ContinuePreviousRunConfig(Config):
    """Configure whether to continue a previous extraction run or not."""

    continue_run: bool = False


@op(out={"continue_run": Out(is_required=False), "new_run": Out(is_required=False)})
def continue_previous_run(config: ContinuePreviousRunConfig):
    """Create branch dictating whether a previous extraction run is continued or not."""
    if config.continue_run:
        yield Output(True, "continue_run")
    else:
        yield Output(True, "new_run")


@op(out={"previous_run_extraction_metadata": Out(), "previous_extracted": Out()})
def merge_branches(dfs: list[tuple[pd.DataFrame, pd.DataFrame]]):
    """Merge branches created by `continue_previous_run` and return."""
    dfs = dfs[0]
    return dfs[0], dfs[1]


@graph(
    out={
        "previous_run_extraction_metadata": GraphOut(),
        "previous_extracted": GraphOut(),
    }
)
def get_starting_data():
    """Get previous run data if configured to do so."""
    continue_run, new_run = continue_previous_run()
    previous_data = get_previous_run_data(continue_run)
    new_data = get_empty_run_data(new_run)
    return merge_branches([previous_data, new_data])


basic_10k_extract_graph = extract_graph_factory("basic_10k", basic_10k.extract)


@pudl_model(
    "basic_10k_extraction", resources={"cloud_interface": cloud_interface_resource}
)
@graph
def basic_10k_extraction_model():
    """Implement basic 10k extraction pudl_model."""
    previous_extraction_metadata, previous_extracted = get_starting_data()
    return basic_10k_extract_graph(previous_extraction_metadata, previous_extracted)


def compute_validation_metrics(
    computed_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    value_col: str,
) -> dict:
    """Compute precision and recall for extraction compared to validation set.

    Arg:
        computed_set: Extracted data.
        validation_set: Expected extraction results.
        value_col: Column to compare when computing metrics.
    """
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
