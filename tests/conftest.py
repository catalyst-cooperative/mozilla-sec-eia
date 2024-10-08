"""PyTest configuration module. Defines useful fixtures, command line args."""

import logging
import os
from pathlib import Path

import pytest

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
def set_test_mlflow_env_vars_factory(tmp_path):
    def factory():
        os.environ["DAGSTER_HOME"] = str(tmp_path)
        # Use in memory tracking backend unless USE_TRACKING_SERVER is set
        if not os.getenv("USE_TRACKING_SERVER"):
            os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///:memory:"

    return factory
