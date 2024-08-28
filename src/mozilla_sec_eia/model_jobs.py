"""Define asset jobs and configuration."""

import logging

import coloredlogs
from dagster import Definitions

from mozilla_sec_eia.library import get_ml_model_jobs

logger = logging.getLogger("catalystcoop")
log_format = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s"
coloredlogs.install(fmt=log_format, logger=logger)

defs = Definitions(
    jobs=get_ml_model_jobs(),
)
