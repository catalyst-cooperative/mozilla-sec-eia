"""Implement base class for an SEC10k extractor."""

import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetOut,
    ConfigurableResource,
    StaticPartitionsDefinition,
    asset,
    multi_asset,
    with_resources,
)

from .utils.cloud import GCSArchive


class Sec10kExtractor(ConfigurableResource):
    """Base class for extracting SEC 10k data."""

    cloud_interface: GCSArchive
    name: str

    def extract_filings(
        self, filing_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Method must be implemented by subclasses to extract SEC10k filings."""
        raise NotImplementedError(
            "extract_filings must be implemented by any subclass!"
        )


# Create year_quarter partitions
year_quarter_partitions = StaticPartitionsDefinition(
    [f"{year}q{quarter}" for year in range(1994, 2024) for quarter in range(1, 5)]
)


@asset(partitions_def=year_quarter_partitions)
def sec10k_filing_metadata(
    context: AssetExecutionContext,
    cloud_interface: GCSArchive,
) -> pd.DataFrame:
    """Return filing metadata for year_quarter partition."""
    year_quarter = context.partition_key
    df = cloud_interface.get_metadata(year_quarter=year_quarter)
    return df


def sec10k_extraction_asset_factory(
    name: str,
    sec10k_extractor: Sec10kExtractor,
    partitions_def=year_quarter_partitions,
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
        required_resource_keys={sec10k_extractor.name},
    )
    def extract_filings(
        context: AssetExecutionContext, sec10k_filing_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run Sec10kExtractor on selected partition and return."""
        extractor = context.resources.original_resource_dict[sec10k_extractor.name]
        extraction_metadata, extracted = extractor.extract_filings(
            sec10k_filing_metadata
        )
        return extraction_metadata, extracted

    return with_resources([extract_filings], {sec10k_extractor.name: sec10k_extractor})[
        0
    ]
