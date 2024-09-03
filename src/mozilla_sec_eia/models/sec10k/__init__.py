"""Implement models to extract data from SEC10k filings."""

from dagster import (
    Definitions,
    load_assets_from_modules,
    load_assets_from_package_module,
)

from mozilla_sec_eia.library import model_jobs
from mozilla_sec_eia.library.mlflow import (
    MlflowInterface,
    mlflow_interface_resource,
    mlflow_validation_io_managers,
)

from . import basic_10k, ex_21, extract
from .utils.cloud import cloud_interface_resource

basic_10k_assets = load_assets_from_modules([basic_10k])
ex21_assets = load_assets_from_package_module(ex_21)
shared_assets = load_assets_from_modules([extract])

basic_10k_production_job = model_jobs.create_production_model_job(
    "basic_10k_extraction",
    basic_10k.production_assets,
)

basic_10k_validation_job = model_jobs.create_production_model_job(
    "basic_10k_extraction_validation",
    basic_10k.validation_assets,
)


ex21_production_job = model_jobs.create_production_model_job(
    "ex21_extraction",
    ex_21.production_assets,
)

ex21_validation_job = model_jobs.create_validation_model_job(
    "ex21_extraction_validation",
    ex_21.validation_assets,
)


defs = Definitions(
    assets=basic_10k_assets + ex21_assets + shared_assets,
    jobs=[
        basic_10k_production_job,
        basic_10k_validation_job,
        ex21_production_job,
        ex21_validation_job,
    ],
    resources={
        "cloud_interface": cloud_interface_resource,
        "mlflow_interface": mlflow_interface_resource,
    }
    | mlflow_validation_io_managers,
)
