"""Unit tests for GCSArchive class."""

import unittest
from dataclasses import dataclass

import pandas as pd
import pytest
from mozilla_sec_eia.utils import GCSArchive


@pytest.fixture
def test_archive():
    """Return test GCSArchive class."""
    with (
        unittest.mock.patch("mozilla_sec_eia.utils.GCSArchive._get_engine"),
        unittest.mock.patch("mozilla_sec_eia.utils.GCSArchive._get_bucket"),
    ):
        return GCSArchive(
            GCS_BUCKET_NAME="bucket_name",
            GCS_METADATA_DB_INSTANCE_CONNECTION="metadata_db_instance_connection",
            GCS_IAM_USER="user",
            GCS_METADATA_DB_NAME="metadata_db_name",
        )


@dataclass
class _FakeBlob:
    name: str


@pytest.mark.parametrize(
    "archive_files,metadata_files,valid",
    [
        (
            [
                _FakeBlob("sec10k/sec10k-1993q1/filing1.txt"),
                _FakeBlob("sec10k/sec10k-1996q2/filing2.txt"),
                _FakeBlob("sec10k/sec10k-2000q4/filing3.txt"),
            ],
            [
                "filing1.txt",
                "filing2.txt",
                "filing3.txt",
            ],
            True,
        ),
        (
            [
                _FakeBlob("sec10k/sec10k-1993q1/filing1.txt"),
                _FakeBlob("sec10k/sec10k-1996q2/filing2.txt"),
                _FakeBlob("sec10k/sec10k-2000q4/filing3.txt"),
                _FakeBlob("sec10k/sec10k-2001q3/filing4.txt"),
            ],
            [
                "filing1.txt",
                "filing2.txt",
                "filing3.txt",
            ],
            False,
        ),
        (
            [
                _FakeBlob("sec10k/sec10k-1993q1/filing1.txt"),
                _FakeBlob("sec10k/sec10k-1996q2/filing2.txt"),
                _FakeBlob("sec10k/sec10k-2000q4/filing3.txt"),
            ],
            [
                "filing1.txt",
                "filing2.txt",
                "filing3.txt",
                "filing4.txt",
            ],
            False,
        ),
    ],
)
def test_validate_archive(test_archive, archive_files, metadata_files, valid, mocker):
    """Test archive validation functionality."""
    test_archive._bucket.list_blobs.return_value = archive_files

    metadata_mock = mocker.MagicMock(
        return_value=pd.DataFrame({"Filename": metadata_files})
    )
    mocker.patch("mozilla_sec_eia.utils.GCSArchive.get_metadata", new=metadata_mock)

    assert test_archive.validate_archive() == valid
