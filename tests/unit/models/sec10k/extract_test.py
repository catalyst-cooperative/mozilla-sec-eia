"""Test extraction tools/methods."""

import logging
from unittest.mock import Mock

import pandas as pd
from dagster import asset, build_asset_context, materialize
from mozilla_sec_eia.models.sec10k.extract import (
    Sec10kExtractor,
    sec10k_extraction_asset_factory,
    sec10k_filing_metadata,
)
from mozilla_sec_eia.models.sec10k.utils.cloud import GCSArchive

logger = logging.getLogger(f"catalystcoop.{__name__}")


def test_sec10k_filing_metadata():
    """Test loading sec10k filing metadata."""
    # Prepare inputs to sec10k_filing_metadata
    context = build_asset_context(partition_key="2024q1")
    cloud_interface = Mock()
    output_df = pd.DataFrame({"col": ["fake_col"]})
    cloud_interface.get_metadata.return_value = output_df

    returned_df = sec10k_filing_metadata(
        context=context,
        cloud_interface=cloud_interface,
    )

    # Check that GCSArchive.get_metadata was called correctly
    cloud_interface.get_metadata.assert_called_once_with(year_quarter="2024q1")
    pd.testing.assert_frame_equal(returned_df, output_df)


def test_sec10k_extraction():
    """Test loading sec10k filing metadata."""
    fake_extraction_metadata = pd.DataFrame({"extraction_metadata": ["fake_col"]})
    fake_extracted = pd.DataFrame({"extracted": ["fake_col"]})
    fake_filing_metadata = pd.DataFrame({"filing_metadata": ["fake_col"]})

    # Create fake Sec10kExtractor
    class TestSec10kExtractor(Sec10kExtractor):
        name: str = "test_extractor"

        def extract_filings(self, filing_metadata):
            pd.testing.assert_frame_equal(filing_metadata, fake_filing_metadata)
            return fake_extraction_metadata, fake_extracted

    # Create fake GCSArchive
    class FakeArchive(GCSArchive):
        filings_bucket_name: str = ""
        labels_bucket_name: str = ""
        metadata_db_instance_connection: str = ""
        user: str = ""
        metadata_db_name: str = ""
        project: str = ""

        def setup_for_execution(self, context):
            pass

    # Asset to return fake filing metadata
    @asset
    def fake_filing_metadata_asset():
        return fake_filing_metadata

    # Create fake extraction asset with configured inputs
    extraction_multi_asset = sec10k_extraction_asset_factory(
        name="test_sec10k_extraction",
        sec10k_extractor=TestSec10kExtractor(cloud_interface=FakeArchive()),
        filing_metadata_asset_name="fake_filing_metadata_asset",
        extracted_asset_name="test_sec10k_extraction",
        extraction_metadata_asset_name="test_sec10k_extraction_metadata",
        partitions_def=None,
    )

    # Run assets and review results
    result = materialize([fake_filing_metadata_asset, extraction_multi_asset])
    pd.testing.assert_frame_equal(
        result.asset_value("test_sec10k_extraction_metadata"), fake_extraction_metadata
    )
    pd.testing.assert_frame_equal(
        result.asset_value("test_sec10k_extraction"), fake_extracted
    )
