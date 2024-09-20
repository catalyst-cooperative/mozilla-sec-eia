"""Validate basic 10k and exhibit 21 extraction."""

import logging
import os
import unittest

import dotenv

from mozilla_sec_eia.library.mlflow.mlflow_resource import (
    _configure_mlflow,
    get_most_recent_run,
)
from mozilla_sec_eia.models import sec10k

logger = logging.getLogger(f"catalystcoop.{__name__}")

# TODO: Make validation tests log to tracking server on merge to main


def test_basic_10k_validation(
    set_test_mlflow_env_vars_factory,
):
    """Test basic_10k_validation_job."""
    dotenv.load_dotenv(override=True)
    set_test_mlflow_env_vars_factory()
    result = sec10k.defs.get_job_def(
        "basic_10k_extraction_validation"
    ).execute_in_process()

    run = get_most_recent_run("basic_10k_extraction_validation", result.run_id)

    assert run.data.metrics["precision"] == 1
    assert run.data.metrics["recall"] == 1


def test_ex21_validation(
    tmp_path,
    set_test_mlflow_env_vars_factory,
):
    """Test ex21_validation_job."""
    dotenv.load_dotenv(override=True)
    _configure_mlflow(
        os.getenv("MLFLOW_TRACKING_URI"),
        os.getenv("GCS_PROJECT"),
    )
    pretrained_model = sec10k.utils.layoutlm._load_pretrained_layoutlm(
        cache_path=tmp_path
    )

    with unittest.mock.patch(
        "mozilla_sec_eia.models.sec10k.utils.layoutlm._load_pretrained_layoutlm",
        new=lambda cache_path, version: pretrained_model,
    ):
        set_test_mlflow_env_vars_factory()
        result = sec10k.defs.get_job_def(
            "ex21_extraction_validation"
        ).execute_in_process()

    run = get_most_recent_run("ex21_extraction_validation", result.run_id)

    assert run.data.metrics["avg_subsidiary_jaccard_sim"] > 0.85
    assert run.data.metrics["avg_location_jaccard_sim"] > 0.9
