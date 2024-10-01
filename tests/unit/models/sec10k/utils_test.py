"""Unit tests for GCSArchive class."""

import io
import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from mozilla_sec_eia.models.sec10k.utils.cloud import (
    Exhibit21,
    GCSArchive,
    Sec10K,
)


@pytest.fixture
def test_archive():
    """Return test GCSArchive class."""
    return GCSArchive()


@dataclass
class _FakePath:
    files: list[str]

    def iterdir(self):
        """Fake iterdir"""
        yield from self.files


@pytest.mark.parametrize(
    "archive_files,metadata_files,valid",
    [
        (
            _FakePath(
                files=[
                    Path("sec10k/sec10k-1993q1/filing1.txt"),
                    Path("sec10k/sec10k-1996q2/filing2.txt"),
                    Path("sec10k/sec10k-2000q4/filing3.txt"),
                ]
            ),
            [
                "filing1.txt",
                "filing2.txt",
                "filing3.txt",
            ],
            True,
        ),
        (
            _FakePath(
                files=[
                    Path("sec10k/sec10k-1993q1/filing1.txt"),
                    Path("sec10k/sec10k-1996q2/filing2.txt"),
                    Path("sec10k/sec10k-2000q4/filing3.txt"),
                    Path("sec10k/sec10k-2001q3/filing4.txt"),
                ]
            ),
            [
                "filing1.txt",
                "filing2.txt",
                "filing3.txt",
            ],
            False,
        ),
        (
            _FakePath(
                files=[
                    Path("sec10k/sec10k-1993q1/filing1.txt"),
                    Path("sec10k/sec10k-1996q2/filing2.txt"),
                    Path("sec10k/sec10k-2000q4/filing3.txt"),
                ]
            ),
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
    with unittest.mock.patch(
        "mozilla_sec_eia.models.sec10k.utils.GCSArchive.filings_bucket_path",
        new=archive_files,
    ):
        metadata_mock = mocker.MagicMock(
            return_value=pd.DataFrame({"filename": metadata_files}).set_index(
                "filename"
            )
        )
        mocker.patch(
            "mozilla_sec_eia.models.sec10k.utils.cloud.GCSArchive.get_metadata",
            new=metadata_mock,
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
    with unittest.mock.patch(
        "mozilla_sec_eia.models.sec10k.utils.cloud.logger"
    ) as mock_logger:
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
