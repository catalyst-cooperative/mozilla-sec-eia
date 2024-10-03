"""Implement models to extract data from SEC10k filings."""

from dagster import (
    Config,
    Definitions,
    define_asset_job,
    file_relative_path,
    in_process_executor,
    load_assets_from_modules,
    load_assets_from_package_module,
)
from dagstermill import (
    ConfigurableLocalOutputNotebookIOManager,
    define_dagstermill_asset,
)
from upath import UPath

from mozilla_sec_eia.library import model_jobs
from mozilla_sec_eia.library.generic_io_managers import PandasParquetIOManager
from mozilla_sec_eia.library.mlflow import (
    MlflowPyfuncModelIOManager,
    mlflow_interface_resource,
    mlflow_train_test_io_managers,
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

basic_10k_validation_job = model_jobs.create_validation_model_job(
    "basic_10k_extraction_validation",
    basic_10k.validation_assets,
)


ex21_production_job = model_jobs.create_production_model_job(
    "ex21_extraction",
    ex_21.production_assets,
    concurrency_limit=4,
)


class TrainConfig(Config):
    """Config for training notebook."""

    uri: str = "runs:/c363159de2f5439c93dd972d51247370/layoutlm_extractor"
    training_set: str = "labeledv0.2"


exhibit21_extractor = define_dagstermill_asset(
    name="exhibit21_extractor",
    notebook_path=file_relative_path(__file__, "notebooks/exhibit21_extractor.ipynb"),
    config_schema=TrainConfig.to_config_schema(),
)
ex21_training_job = define_asset_job(
    "ex21_training",
    selection=[exhibit21_extractor],
    executor_def=in_process_executor,
)


defs = Definitions(
    assets=basic_10k_assets + ex21_assets + shared_assets + [exhibit21_extractor],
    jobs=[
        basic_10k_production_job,
        basic_10k_validation_job,
        ex21_production_job,
        ex21_training_job,
    ],
    resources={
        "cloud_interface": cloud_interface_resource,
        "mlflow_interface": mlflow_interface_resource,
        "layoutlm_io_manager": MlflowPyfuncModelIOManager(
            mlflow_interface=mlflow_interface_resource,
            uri="runs:/b959cfa0ba3c4b91a0f8fe158cd0109f/exhibit21_extractor",
        ),
        "pandas_parquet_io_manager": PandasParquetIOManager(
            base_path=UPath("gs://sec10k-outputs")
        ),
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
    }
    | mlflow_train_test_io_managers,
)
