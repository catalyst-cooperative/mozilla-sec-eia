"""Test extraction tools/methods."""

import logging
from unittest.mock import Mock

import dagster
import pandas as pd
import pytest
from dagster import materialize

from mozilla_sec_eia.models.sec10k.extract import sec10k_filing_metadata

logger = logging.getLogger(f"catalystcoop.{__name__}")


def test_sec10k_filing_metadata():
    """Test loading sec10k filing metadata."""
    fake_cloud_interface = Mock()
    df = pd.DataFrame({"fake_df": ["fake_col"]})
    fake_cloud_interface.get_metadata.return_value = df
    output = materialize(
        [sec10k_filing_metadata],
        partition_key="2020q1",
        resources={"cloud_interface": fake_cloud_interface},
    )

    fake_cloud_interface.get_metadata.assert_called_once_with(year_quarter="2020q1")
    pd.testing.assert_frame_equal(df, output.asset_value("sec10k_filing_metadata"))


def test_sec10k_filing_metadata_bad_return_type():
    """Test loading sec10k_filing_metadata with bad return type."""
    fake_cloud_interface = Mock()
    fake_cloud_interface.get_metadata.return_value = "should be DataFrame"
    with pytest.raises(dagster._core.errors.DagsterTypeCheckDidNotPass):
        materialize(
            [sec10k_filing_metadata],
            partition_key="2020q1",
            resources={"cloud_interface": fake_cloud_interface},
        )
