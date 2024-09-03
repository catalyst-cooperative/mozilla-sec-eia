"""Define jobs to test/validate PUDL models."""

import logging

import coloredlogs
from dagster import Definitions

from mozilla_sec_eia.library.mlflow import (
    get_mlflow_io_manager,
    mlflow_train_test_interface,
)
from mozilla_sec_eia.library.pipeline import (
    PUDL_PIPELINE_VALIDATION_ASSETS,
    PUDL_PIPELINE_VALIDATION_JOBS,
    PUDL_PIPELINE_VALIDATION_RESOURCES,
)

logger = logging.getLogger("catalystcoop")
log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
coloredlogs.install(fmt=log_format, logger=logger)

validation_io_resources = {
    key: get_mlflow_io_manager(
        key, mlflow_interface=mlflow_train_test_interface, pandas_file_type="csv"
    )
    for key in ["mlflow_pandas_artifact_io_manager", "mlflow_metrics_io_manager"]
} | PUDL_PIPELINE_VALIDATION_RESOURCES

validation_pipelines = Definitions(
    assets=PUDL_PIPELINE_VALIDATION_ASSETS,
    jobs=PUDL_PIPELINE_VALIDATION_JOBS,
    resources=validation_io_resources
    | {"mlflow_interface": mlflow_train_test_interface},
)
