"""PyTest configuration module. Defines useful fixtures, command line args."""

import logging
from pathlib import Path

import mlflow
import pytest
from mozilla_sec_eia.library.mlflow import MlflowInterface

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


class TestTracker(MlflowInterface):
    """Create sub-class of `MlflowInterface` to use in testing context.

    Test class creates an in-memory sqlite db for tracking, and a temporary directory
    for artifact storage.
    """

    def _get_tracking_password(self):
        return "password"


@pytest.fixture
def test_tracker_factory(tmp_path):
    def factory(experiment_name: str) -> TestTracker:
        return TestTracker(
            artifact_location=str(tmp_path),
            tracking_uri="sqlite:///:memory:",
            experiment_name=experiment_name,
            project="",
        )

    return factory


@pytest.fixture
def get_most_recent_mlflow_run_factory():
    def _get_run(experiment_name: str):
        """Search mlflow for most recent run with specified experiment name."""
        run_metadata = mlflow.search_runs(
            experiment_names=[experiment_name],
        )

        # Mlflow returns runs ordered by their runtime, so it's easy to grab the latest run
        # This assert will ensure this doesn't silently break if the ordering changes
        assert run_metadata.loc[0, "end_time"] == run_metadata["end_time"].max()
        return mlflow.get_run(run_metadata.loc[0, "run_id"])

    return _get_run
