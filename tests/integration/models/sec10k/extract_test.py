"""Validate basic 10k and exhibit 21 extraction."""

import logging

import dotenv
import pytest
from mozilla_sec_eia.models import sec10k

logger = logging.getLogger(f"catalystcoop.{__name__}")


def test_basic_10k_validation(
    test_tracker_factory,
    get_most_recent_mlflow_run_factory,
):
    """Test basic_10k_validation_job."""
    dotenv.load_dotenv()
    sec10k.defs.get_job_def("basic_10k_extraction_validation").execute_in_process()

    run = get_most_recent_mlflow_run_factory("basic_10k_extraction_validation")

    assert run.data.metrics["precision"] == 1
    assert run.data.metrics["recall"] == 1


@pytest.mark.xfail
def test_ex21_validation(
    test_tracker_factory,
    get_most_recent_mlflow_run_factory,
):
    """Test ex21_validation_job."""
    dotenv.load_dotenv()
    sec10k.defs.get_job_def("ex21_extraction_validation").execute_in_process()

    run = get_most_recent_mlflow_run_factory("ex21_extraction_validation")

    assert run.data.metrics["avg_subsidiary_jaccard_sim"] > 0.85
    assert run.data.metrics["avg_location_jaccard_sim"] > 0.9
