"""Implement models to extract data from SEC10k filings."""

from dagster import (
    AssetIn,
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
ex21_data_assets = load_assets_from_modules([ex_21.data])
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
    tag_concurrency_limits=[
        {"key": "model", "value": "exhibit21_extractor", "limit": 2},
    ],
)


exhibit21_extractor = define_dagstermill_asset(
    name="exhibit21_extractor",
    notebook_path=file_relative_path(__file__, "notebooks/exhibit21_extractor.ipynb"),
    config_schema=ex_21.data.Ex21TrainConfig.to_config_schema(),
    ins={
        "ex21_training_data": AssetIn(),
        "ex21_validation_set": AssetIn(),
        "ex21_failed_parsing_metadata": AssetIn(),
        "ex21_inference_dataset": AssetIn(),
    },
    save_notebook_on_failure=True,
)
ex21_training_job = define_asset_job(
    "ex21_training",
    selection=[exhibit21_extractor] + ex_21.data.ex21_extraction_training_assets,
    executor_def=in_process_executor,
)


exhibit21_layout_classifier = define_dagstermill_asset(
    name="exhibit21_layout_classifier",
    notebook_path=file_relative_path(
        __file__, "notebooks/exhibit21_layout_classifier.ipynb"
    ),
    config_schema=ex_21.data.Ex21TrainConfig.to_config_schema(),
    ins={
        "ex21_layout_labels": AssetIn(),
        "ex21_layout_classifier_training_dataset": AssetIn(),
    },
    save_notebook_on_failure=True,
)
ex21_layout_classifier_training_job = define_asset_job(
    "ex21_layout_classifier_training",
    selection=[exhibit21_layout_classifier] + ex_21.data.ex21_layout_classifier_assets,
    executor_def=in_process_executor,
)


defs = Definitions(
    assets=basic_10k_assets
    + ex21_assets
    + shared_assets
    + [exhibit21_extractor, exhibit21_layout_classifier]
    + ex21_data_assets,
    jobs=[
        basic_10k_production_job,
        basic_10k_validation_job,
        ex21_production_job,
        ex21_training_job,
        ex21_layout_classifier_training_job,
    ],
    resources={
        "cloud_interface": cloud_interface_resource,
        "mlflow_interface": mlflow_interface_resource,
        "layoutlm_io_manager": MlflowPyfuncModelIOManager(
            mlflow_interface=mlflow_interface_resource,
            uri="runs:/426dd1b67cbd4677b6fa22b6b9d9173a/exhibit21_extractor",
        ),
        "ex21_classifier_io_manager": MlflowPyfuncModelIOManager(
            mlflow_interface=mlflow_interface_resource,
            uri="runs:/cbdd906766b2427c93e9c957be6ea9c8/exhibit21_layout_classifier",
        ),
        "pandas_parquet_io_manager": PandasParquetIOManager(
            base_path=UPath("gs://sec10k-outputs/v2")
        ),
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
    }
    | mlflow_train_test_io_managers,
)
