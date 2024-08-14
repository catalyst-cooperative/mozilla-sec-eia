"""Validate basic 10k and exhibit 21 extraction."""

import dotenv
from dagster import EnvVar, build_asset_context
from mozilla_sec_eia.extract import (
    _get_most_recent_run,
    basic_10k_validate,
)
from mozilla_sec_eia.utils.cloud import GCSArchive, MlflowInterface


def test_basic_10k_extraction():
    """Run full 10k extraction on validation set and verify desired metrics are met."""
    dotenv.load_dotenv()
    experiment_name = "basic_10k_validate_test"
    cloud_interface = GCSArchive(
        filings_bucket_name=EnvVar("GCS_FILINGS_BUCKET_NAME"),
        labels_bucket_name=EnvVar("GCS_LABELS_BUCKET_NAME"),
        metadata_db_instance_connection=EnvVar("GCS_METADATA_DB_INSTANCE_CONNECTION"),
        user=EnvVar("GCS_IAM_USER"),
        metadata_db_name=EnvVar("GCS_METADATA_DB_NAME"),
        project=EnvVar("GCS_PROJECT"),
    )

    with build_asset_context(
        resources={
            "basic_10k_extract_validate_mlflow": MlflowInterface(
                experiment_name=experiment_name,
                continue_run=False,
                tracking_uri="sqlite:///:memory:",
                cloud_interface=cloud_interface,
            ),
            "cloud_interface": cloud_interface,
        }
    ) as context:
        basic_10k_validate(context)
    run = _get_most_recent_run(experiment_name)

    assert run.data.metrics["precision"] == 1
    assert run.data.metrics["recall"] == 1
