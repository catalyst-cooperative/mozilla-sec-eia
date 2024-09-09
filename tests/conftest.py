"""PyTest configuration module. Defines useful fixtures, command line args."""

import logging
import unittest
from pathlib import Path

import pytest

from mozilla_sec_eia.utils.cloud import GoogleCloudSettings, initialize_mlflow

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add package-specific command line options to pytest.

    This is slightly magical -- pytest has a hook that will run this function
    automatically, adding any options defined here to the internal pytest options that
    already exist.
    """
    parser.addoption(
        "--sandbox",
        action="store_true",
        default=False,
        help="Flag to indicate that the tests should use a sandbox.",
    )


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture
def test_settings(test_dir):
    """Return test GoogleCloudSettings object."""
    return GoogleCloudSettings(_env_file=test_dir / "test.env")


@pytest.fixture
def test_mlflow_init_func(test_settings):
    """Return a function that can replace ``initialize_mlflow`` with no external calls."""

    def _test_init():
        with unittest.mock.patch(
            "mozilla_sec_eia.utils.cloud._access_secret_version",
            new=lambda *args: "password",
        ):
            return initialize_mlflow(test_settings)

    return _test_init
