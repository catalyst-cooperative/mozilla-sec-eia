"""Implements shared tooling for machine learning models in PUDL."""

from . import models


def get_ml_pipeline_jobs() -> list[str]:
    """Return all jobs created through `pudl_model` decorator."""
    return list(models.PUDL_PIPELINES.values())
