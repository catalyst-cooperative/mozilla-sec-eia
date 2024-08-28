"""Define asset jobs and configuration."""

import logging

import coloredlogs
from dagster import Definitions

from mozilla_sec_eia.library import ml_tools

logger = logging.getLogger("catalystcoop")
log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
coloredlogs.install(fmt=log_format, logger=logger)

defs = Definitions(
    jobs=ml_tools.get_ml_model_jobs(),
    resources=ml_tools.get_ml_model_resources(),
)
