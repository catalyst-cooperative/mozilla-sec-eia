"""Implement top level extraction methods and tooling."""

import io
import logging
import math
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pandera as pa
from dagster import (
    Config,
    DynamicOut,
    DynamicOutput,
    GraphDefinition,
    OpDefinition,
    graph,
    op,
)
from mlflow.entities import Run

from mozilla_sec_eia import basic_10k
from mozilla_sec_eia.utils.cloud import GCSArchive
from mozilla_sec_eia.utils.ml_tools.experiment_tracking import (
    get_tracking_resource_name,
)
from mozilla_sec_eia.utils.ml_tools.models import pudl_model

logger = logging.getLogger(f"catalystcoop.{__name__}")

DATASETS = ["ex21", "basic_10k"]


class ExtractionMetadataSchema(pa.DataFrameModel):
    """Define the required schema for extraction metadata.

    Extra columns are permitted, but these are required for computing extraction metrics.
    """

    filename: pa.typing.Index[str] = pa.Field(check_name=True)
    success: bool = pa.Field(coerce=True)


def _load_artifact_as_csv(run: Run, artifact_name: str) -> pd.DataFrame:
    """Download a CSV and parse to DataFrame from mlflow tracking server."""
    df = pd.read_csv(
        io.StringIO(mlflow.artifacts.load_text(run.info.artifact_uri + artifact_name))
    )
    return df


def _log_artifact_as_csv(
    artifact: pd.DataFrame, artifact_name: str, index: bool = True
):
    """Upload a DataFrame as a CSV to mlflow tracking server."""
    return mlflow.log_text(artifact.to_csv(index=index), artifact_name)


def _load_artifact_as_parquet(run: Run, artifact_name: str) -> pd.DataFrame:
    """Download a CSV and parse to DataFrame from mlflow tracking server."""
    df = pd.read_parquet(run.info.artifact_uri + artifact_name)
    return df


def _log_artifact_as_parquet(
    artifact: pd.DataFrame, artifact_name: str, index: bool = True
):
    """Upload a DataFrame as a CSV to mlflow tracking server."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_path = Path(tmp_dir) / artifact_name
        artifact.to_parquet(parquet_path, index=index)
        return mlflow.log_artifact(parquet_path, artifact_name)


@op
def get_filings_to_extract(
    cloud_interface: GCSArchive,
) -> pd.DataFrame:
    """Return filing metadata."""
    return cloud_interface.get_metadata()


class ChunkFilingsConfig(Config):
    """Config how many filings are extracted and chunk_size for extraction."""

    chunk_size: int = 1000
    num_filings: int = -1


@op(out=DynamicOut())
def chunk_filings(
    config: ChunkFilingsConfig,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Split filings into chunks for parallel extraction."""
    filings_to_extract = metadata
    if config.num_filings > 0:
        filings_to_extract = filings_to_extract.sample(config.num_filings)

    for i, chunk in enumerate(
        np.array_split(
            filings_to_extract, math.ceil(len(filings_to_extract) / config.chunk_size)
        )
    ):
        yield DynamicOutput(chunk, mapping_key=str(i))


def extract_model_factory(
    dataset_name: str, extract_op: OpDefinition | GraphDefinition
):
    """Produce a `pudl_model` to extract data from sec10k filings."""
    experiment_name = f"{dataset_name}_extraction"
    experiment_tracker_resource = get_tracking_resource_name(experiment_name)

    @op(required_resource_keys=[experiment_tracker_resource])
    def log_extraction_data(
        metadata: pd.DataFrame,
        extraction_metadata: list[pd.DataFrame],
        extracted: list[pd.DataFrame],
    ):
        extraction_metadata = pd.concat(extraction_metadata)
        extracted = pd.concat(extracted)
        # Use metadata to log generic metrics
        extraction_metadata = ExtractionMetadataSchema.validate(extraction_metadata)
        mlflow.log_metrics(
            {
                "num_failed": (~extraction_metadata["success"]).sum(),
                "ratio_extracted": len(extraction_metadata) / len(metadata),
            }
        )

        # Log the extraction results + metadata for future reference/analysis
        _log_artifact_as_csv(extraction_metadata, "extraction_metadata.csv")
        _log_artifact_as_parquet(extracted, "extracted.parquet")

    @pudl_model(experiment_name=experiment_name)
    @graph(name=experiment_name)
    def extract_filings():
        filings_to_extract = get_filings_to_extract()
        filing_chunks = chunk_filings(filings_to_extract)
        extraction_metadata, extracted = filing_chunks.map(extract_op)

        return log_extraction_data(
            filings_to_extract,
            extraction_metadata.collect(),
            extracted.collect(),
        )

    return extract_filings


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


basic_10k_extract = extract_model_factory("basic_10k", basic_10k.extract)
