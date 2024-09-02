"""Define production pipelines for running PUDL models."""

import logging

import coloredlogs
from dagster import Definitions, EnvVar

from mozilla_sec_eia.library.mlflow import MlflowInterface
from mozilla_sec_eia.library.pipeline import (
    PUDL_PIPELINE_PRODUCTION_ASSETS,
    PUDL_PIPELINE_PRODUCTION_JOBS,
    PUDL_PIPELINE_PRODUCTION_RESOURCES,
)

logger = logging.getLogger("catalystcoop")
log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
coloredlogs.install(fmt=log_format, logger=logger)


mlflow_interface = MlflowInterface(
    experiment_name="",
    tracking_uri=EnvVar("MLFLOW_TRACKING_URI"),
    project=EnvVar("GCS_PROJECT"),
)

production_io_resources = {} | PUDL_PIPELINE_PRODUCTION_RESOURCES

production_pipelines = Definitions(
    assets=PUDL_PIPELINE_PRODUCTION_ASSETS,
    jobs=PUDL_PIPELINE_PRODUCTION_JOBS,
    resources=production_io_resources | {"mlflow_interface": mlflow_interface},
)
