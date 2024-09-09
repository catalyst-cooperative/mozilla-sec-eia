"""Unit tests for GCSArchive class."""

import io
import unittest
from dataclasses import dataclass

import pandas as pd
import pytest

from mozilla_sec_eia.utils.cloud import (
    Exhibit21,
    GCSArchive,
    GoogleCloudSettings,
    Sec10K,
)


@pytest.fixture
def test_archive():
    """Return test GCSArchive class."""
    with (
        unittest.mock.patch("mozilla_sec_eia.utils.cloud.GCSArchive._get_engine"),
        unittest.mock.patch("mozilla_sec_eia.utils.cloud.GCSArchive._get_bucket"),
    ):
        return GCSArchive(
            settings=GoogleCloudSettings(
                GCS_FILINGS_BUCKET_NAME="filings_bucket_name",
                GCS_LABELS_BUCKET_NAME="labels_bucket_name",
                GCS_METADATA_DB_INSTANCE_CONNECTION="metadata_db_instance_connection",
                GCS_IAM_USER="user",
                GCS_METADATA_DB_NAME="metadata_db_name",
                GCS_PROJECT="project_name",
                MLFLOW_TRACKING_URI="http://tracking.server",
            )
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
    test_archive._filings_bucket.list_blobs.return_value = archive_files

    metadata_mock = mocker.MagicMock(
        return_value=pd.DataFrame({"filename": metadata_files})
    )
    mocker.patch(
        "mozilla_sec_eia.utils.cloud.GCSArchive.get_metadata", new=metadata_mock
    )

    assert test_archive.validate_archive() == valid


@pytest.mark.parametrize(
    "filing_text,ex_21_version,actually_has_ex_21",
    [
        (
            """
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<DESCRIPTION>10-K
<TEXT>
<html>
<body>Some text here</body>
</html>
</TEXT>
<DOCUMENT>
<TYPE>EX-21.1
<SEQUENCE>4
<html>
<body>Exhibit 21 text</body>
</html>
</DOCUMENT>
<TEXT>
<html>
<body>Some other text here</body>
</html>
</DOCUMENT>

            """,
            "21.1",
            True,
        ),
        (
            """
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<DESCRIPTION>10-K
<TEXT>
<html>
<body>Some text here</body>
</html>
</TEXT>
<TEXT>
<html>
<body>Some other text here</body>
</html>
</DOCUMENT>

            """,
            "21.1",
            False,
        ),
        (
            """
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<DESCRIPTION>10-K
<TEXT>
<html>
<body>Some text here</body>
</html>
</TEXT>
<TEXT>
<html>
<body>Some other text here</body>
</html>
</DOCUMENT>

            """,
            None,
            False,
        ),
    ],
)
def test_10k(filing_text, ex_21_version, actually_has_ex_21):
    """Test that SEC10k's are properly parsed."""
    with unittest.mock.patch("mozilla_sec_eia.utils.cloud.logger") as mock_logger:
        filing = Sec10K.from_file(
            file=io.StringIO(filing_text),
            filename="sec10k.html",
            cik=0,
            year_quarter="2024q1",
            ex_21_version=ex_21_version,
        )

        match (ex_21_version, actually_has_ex_21):
            case (str(ex_21_version), False):
                mock_logger.warning.assert_called_once_with(
                    "Failed to extract exhibit 21 from sec10k.html"
                )
            case (str(ex_21_version), True):
                assert isinstance(filing.ex_21, Exhibit21)
            case (None, _):
                assert filing.ex_21 is None
