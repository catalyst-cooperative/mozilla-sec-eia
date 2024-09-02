"""Implement top level extraction methods and tooling."""

import logging

import pandas as pd
import pandera as pa
from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetOut,
    StaticPartitionsDefinition,
    asset,
    multi_asset,
    with_resources,
)

from mozilla_sec_eia.library.mlflow import validation
from mozilla_sec_eia.library.pipeline import (
    create_production_pipeline,
    create_validation_pipeline,
)

from .basic_10k import Basic10kExtractor
from .extract import Sec10kExtractor
from .utils.cloud import GCSArchive, cloud_interface_resource

logger = logging.getLogger(f"catalystcoop.{__name__}")

DATASETS = ["ex21", "basic_10k"]


class ExtractionMetadataSchema(pa.DataFrameModel):
    """Define the required schema for extraction metadata.

    Extra columns are permitted, but these are required for computing extraction metrics.
    """

    filename: pa.typing.Index[str] = pa.Field(check_name=True)
    success: bool = pa.Field(coerce=True)


# Create year_quarter partitions
partitions_def = StaticPartitionsDefinition(
    [f"{year}q{quarter}" for year in range(1994, 2024) for quarter in range(1, 5)]
)


def sec10k_extraction_asset_factory(
    name: str,
    sec10k_extractor: Sec10kExtractor,
    partitions_def=None,
    filing_metadata_asset_name: str = "sec10k_filing_metadata",
    extraction_metadata_asset_name: str = "extraction_metadata",
    extracted_asset_name: str = "extraction_metadata",
):
    """Create asset to extract data from sec10k data.

    Args:
        name: Name of extraction asset.
        sec10k_extractor: Subclass of Sec10kExtractor used to extract data.
        partitions_def: Partitions for asset (production uses year_quarter parts,
            validation is not partitioned.
        filing_metadata_asset_name: Name of input asset with metadata of filings to
            extract.
        extraction_metadata_asset_name: Name of output asset containing metadata
            from extraction run.
        extracted_asset_name: Name of output asset containing extracted data.
    """

    @multi_asset(
        name=name,
        outs={
            extraction_metadata_asset_name: AssetOut(),
            extracted_asset_name: AssetOut(),
        },
        ins={"sec10k_filing_metadata": AssetIn(filing_metadata_asset_name)},
        partitions_def=partitions_def,
    )
    def extract_filings(
        sec10k_extractor: Sec10kExtractor, sec10k_filing_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run Sec10kExtractor on selected partition and return."""
        extraction_metadata, extracted = sec10k_extractor.extract_filings(
            sec10k_filing_metadata
        )
        return extraction_metadata, extracted

    return with_resources([extract_filings], {"sec10k_extractor": sec10k_extractor})[0]


@asset(partitions_def=partitions_def)
def sec10k_filing_metadata(
    context: AssetExecutionContext,
    cloud_interface: GCSArchive,
) -> pd.DataFrame:
    """Return filing metadata for year_quarter partition."""
    year_quarter = context.partition_key
    df = cloud_interface.get_metadata(year_quarter=year_quarter)
    return df


# Create asset to load basic 10k validation data
basic_10k_validation_set = validation.load_validation_data_asset_factory(
    "basic_10k_validation_set",
    "basic_10k_labels.csv",
    index_cols=["filename", "filer_count", "block", "block_count", "key"],
)


# Create asset to compute precision/recall on basic 10k extraction of validation set
basic_10k_extracted_validation_asset_name = "basic_10k_company_info_validation"
basic_10k_extraction_validation_metrics = (
    validation.pandas_precision_recall_asset_factory(
        validation_asset="basic_10k_validation_set",
        computed_asset=basic_10k_extracted_validation_asset_name,
        value_col="value",
    )
)


@asset(name="sec10k_filing_metadata_validation")
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


# Register basic 10k extraction pipeline
create_production_pipeline(
    "basic_10k_extraction",
    [
        sec10k_filing_metadata,
        sec10k_extraction_asset_factory(
            "basic_10k",
            Basic10kExtractor(cloud_interface=cloud_interface_resource),
            partitions_def=partitions_def,
            extraction_metadata_asset_name="basic_10k_extraction_metadata",
            extracted_asset_name="basic_10k_company_info",
        ),
    ],
    resources={"cloud_interface": cloud_interface_resource},
)


# Register basic 10k extraction validation pipeline
create_validation_pipeline(
    "basic_10k_extraction",
    [
        basic_10k_validation_filing_metadata,
        sec10k_extraction_asset_factory(
            "basic_10k",
            Basic10kExtractor(cloud_interface=cloud_interface_resource),
            filing_metadata_asset_name="sec10k_filing_metadata_validation",
            extraction_metadata_asset_name="basic_10k_extraction_validation_metadata",
            extracted_asset_name=basic_10k_extracted_validation_asset_name,
        ),
        basic_10k_validation_set,
        basic_10k_extraction_validation_metrics,
    ],
    resources={"cloud_interface": cloud_interface_resource},
)
