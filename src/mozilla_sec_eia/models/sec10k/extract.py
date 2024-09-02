"""Implement base class for an SEC10k extractor."""

import pandas as pd
from dagster import ConfigurableResource

from .utils.cloud import GCSArchive


class Sec10kExtractor(ConfigurableResource):
    """Base class for extracting SEC 10k data."""

    cloud_interface: GCSArchive

    def extract_filings(
        self, filing_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Method must be implemented by subclasses to extract SEC10k filings."""
        raise NotImplementedError(
            "extract_filings must be implemented by any subclass!"
        )
