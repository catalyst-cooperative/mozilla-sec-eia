"""Tools for constructing datasets used by exhibit 21 extraction model."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from dagster import AssetOut, Config, asset, multi_asset

from mozilla_sec_eia.library import validation_helpers

from ...entities import ex21_extract_type, sec10k_extract_metadata_type
from ...utils.cloud import GCSArchive
from ..ex21_validation_helpers import clean_ex21_validation_set
from .inference import create_inference_dataset
from .training import format_as_ner_annotations


class Ex21TrainingConfig(Config):
    """Configure asset to produce ex21 training data."""

    training_set: str = "labeledv0.2"


@asset
def ex21_training_data(config: Ex21TrainingConfig):
    """Construct training dataset for ex 21 extraction."""
    with TemporaryDirectory() as temp_dir:
        ner_annotations = format_as_ner_annotations(
            labeled_json_path=Path(temp_dir) / "sec10k_filings" / "labeled_jsons",
            pdfs_path=Path(temp_dir) / "sec10k_filings" / "pdfs",
            gcs_folder_name=config.training_set,
        )
    return ner_annotations


@asset(dagster_type=ex21_extract_type)
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
        filing_metadata.index.isin(ex21_validation_set["filename"].unique())
    ]


@multi_asset(
    outs={
        "ex21_failed_parsing_metadata": AssetOut(
            dagster_type=sec10k_extract_metadata_type,
        ),
        "ex21_inference_dataset": AssetOut(),
    },
)
def ex21_inference_dataset(
    cloud_interface: GCSArchive,
    ex21_validation_filing_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct inference dataset for ex 21 extraction."""
    return create_inference_dataset(
        filing_metadata=ex21_validation_filing_metadata,
        cloud_interface=cloud_interface,
    )
