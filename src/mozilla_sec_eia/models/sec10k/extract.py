"""Implement base class for an SEC10k extractor."""

import pandas as pd
from dagster import (
    AssetExecutionContext,
    StaticPartitionsDefinition,
    asset,
)

from .utils.cloud import GCSArchive

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
