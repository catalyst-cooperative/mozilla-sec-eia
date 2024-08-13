"""Define asset jobs and configuration."""

import logging

import coloredlogs
from dagster import Definitions, EnvVar, define_asset_job

from mozilla_sec_eia.extract import ExtractConfig, basic_10k_extract, basic_10k_validate
from mozilla_sec_eia.utils.cloud import GCSArchive, MlflowInterface

logger = logging.getLogger("catalystcoop")
log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
coloredlogs.install(fmt=log_format, logger=logger)

extract_job = define_asset_job(
    name="extract_job",
    selection=[basic_10k_extract],
)
validate_job = define_asset_job(
    name="validate_job",
    selection=[basic_10k_validate],
)

cloud_interface = GCSArchive(
    filings_bucket_name=EnvVar("GCS_FILINGS_BUCKET_NAME"),
    labels_bucket_name=EnvVar("GCS_LABELS_BUCKET_NAME"),
    metadata_db_instance_connection=EnvVar("GCS_METADATA_DB_INSTANCE_CONNECTION"),
    user=EnvVar("GCS_IAM_USER"),
    metadata_db_name=EnvVar("GCS_METADATA_DB_NAME"),
    project=EnvVar("GCS_PROJECT"),
)

defs = Definitions(
    assets=[basic_10k_validate, basic_10k_extract],
    jobs=[extract_job, validate_job],
    resources={
        "cloud_interface": cloud_interface,
        "basic_10k_extract_config": ExtractConfig(),
        "basic_10k_extract_mlflow": MlflowInterface(
            experiment_name="basic_10k_extraction",
            tracking_uri=EnvVar("MLFLOW_TRACKING_URI"),
            cloud_interface=cloud_interface,
        ),
        "basic_10k_extract_validate_mlflow": MlflowInterface(
            experiment_name="basic_10k_extraction_validation",
            tracking_uri=EnvVar("MLFLOW_TRACKING_URI"),
            cloud_interface=cloud_interface,
        ),
    },
)
