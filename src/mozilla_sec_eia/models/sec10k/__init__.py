"""Implement models to extract data from SEC10k filings."""

from dagster import (
    Definitions,
    load_assets_from_modules,
    load_assets_from_package_module,
)
from upath import UPath

from mozilla_sec_eia.library import model_jobs
from mozilla_sec_eia.library.generic_io_managers import PandasParquetIOManager
from mozilla_sec_eia.library.mlflow import (
    mlflow_interface_resource,
    mlflow_train_test_io_managers,
)

from . import basic_10k, ex_21, extract
from .utils.cloud import cloud_interface_resource
from .utils.layoutlm import LayoutlmIOManager

basic_10k_assets = load_assets_from_modules([basic_10k])
ex21_assets = load_assets_from_package_module(ex_21)
layoutlm_assets = load_assets_from_modules([ex_21.train_extractor])
shared_assets = load_assets_from_modules([extract])

basic_10k_production_job = model_jobs.create_production_model_job(
    "basic_10k_extraction",
    basic_10k.production_assets,
)

basic_10k_validation_job = model_jobs.create_validation_model_job(
    "basic_10k_extraction_validation",
    basic_10k.validation_assets,
)


ex21_production_job = model_jobs.create_production_model_job(
    "ex21_extraction",
    ex_21.production_assets,
    concurrency_limit=4,
)

ex21_validation_job = model_jobs.create_validation_model_job(
    "ex21_extraction_validation",
    ex_21.validation_assets,
)

ex21_test_job = model_jobs.create_validation_model_job(
    "ex21_test", [ex_21.test_extraction_metrics]
)

layoutlm_finetune_job = model_jobs.create_training_job(
    "layoutlm_finetune",
    layoutlm_assets,
)


defs = Definitions(
    assets=basic_10k_assets + ex21_assets + shared_assets + layoutlm_assets,
    jobs=[
        basic_10k_production_job,
        basic_10k_validation_job,
        ex21_production_job,
        ex21_validation_job,
        ex21_test_job,
        layoutlm_finetune_job,
    ],
    resources={
        "cloud_interface": cloud_interface_resource,
        "mlflow_interface": mlflow_interface_resource,
        "layoutlm_io_manager": LayoutlmIOManager(
            mlflow_interface=mlflow_interface_resource
        ),
        "pandas_parquet_io_manager": PandasParquetIOManager(
            base_path=UPath("gs://sec10k-outputs")
        ),
        "exhibit21_extractor": ex_21.exhibit_21_extractor_resource,
    }
    | mlflow_train_test_io_managers,
)
