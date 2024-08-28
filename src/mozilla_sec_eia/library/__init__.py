"""Implements shared tooling for machine learning models in PUDL."""

from . import models


def get_ml_model_resources():
    """Return default configuration for all PUDL models."""
    return models.MODEL_RESOURCES


def get_ml_model_jobs() -> list[str]:
    """Return all jobs created through `pudl_model` decorator."""
    return list(models.PUDL_MODELS.values())
