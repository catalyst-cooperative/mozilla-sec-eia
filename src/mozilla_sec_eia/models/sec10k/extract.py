"""Implement base class for an SEC10k extractor."""

import math

import numpy as np
import pandas as pd
from dagster import (
    AssetExecutionContext,
    Config,
    DynamicOut,
    DynamicOutput,
    StaticPartitionsDefinition,
    asset,
    op,
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


class ChunkFilingsConfig(Config):
    """Set chunk size for chunk_filings."""

    chunk_size: int = 128


@op(out=DynamicOut())
def chunk_filings(config: ChunkFilingsConfig, filings: pd.DataFrame) -> pd.DataFrame:
    """Split filings into chunks for parallel processing."""
    for i, filing_chunk in enumerate(
        np.array_split(filings, math.ceil(len(filings) / config.chunk_size))
    ):
        yield DynamicOutput(filing_chunk, mapping_key=str(i))
