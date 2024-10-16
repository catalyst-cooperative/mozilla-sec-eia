"""Validate basic 10k and exhibit 21 extraction."""

import logging
import os

import dotenv
import mlflow
from dagster import materialize_to_memory

from mozilla_sec_eia.library.mlflow import configure_mlflow
from mozilla_sec_eia.library.mlflow.mlflow_resource import get_most_recent_run
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
    configure_mlflow(
        os.getenv("MLFLOW_TRACKING_URI"),
        os.getenv("GCS_PROJECT"),
    )

    # Load validation data
    result = materialize_to_memory(
        [
            sec10k.ex_21.data.ex21_validation_set,
            sec10k.ex_21.data.ex21_validation_filing_metadata,
            sec10k.ex_21.data.ex21_inference_dataset,
        ],
        resources={"cloud_interface": sec10k.utils.GCSArchive()},
    )
    ex21_inference_dataset = result.output_for_node(
        "ex21_inference_dataset", output_name="ex21_inference_dataset"
    )
    ex21_validation_set = result.output_for_node("ex21_validation_set")

    # Load latest version of pretrained model
    pretrained_model = mlflow.pyfunc.load_model("models:/exhibit21_extractor/latest")
    _, extracted = pretrained_model.predict(ex21_inference_dataset)

    _, _, _, metrics = sec10k.ex_21.ex21_validation_helpers.ex21_validation_metrics(
        extracted, ex21_validation_set
    )

    assert metrics["avg_subsidiary_jaccard_sim"] > 0.85
    assert metrics["avg_location_jaccard_sim"] > 0.83
