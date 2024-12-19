"""Implement models to extract data from SEC10k filings."""

from dagster import (
    AssetIn,
    Definitions,
    Field,
    String,
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
from mozilla_sec_eia.library.generic_io_managers import (
    PandasParquetIOManager,
    PickleUPathIOManager,
)
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


exhibit21_production_job = model_jobs.create_production_model_job(
    "exhibit21_extraction",
    ex_21.production_assets,
    tag_concurrency_limits=[
        {"key": "model", "value": "exhibit21_extractor", "limit": 2},
    ],
    description="Run exhibit 21 extraction pipeline on archived filings.",
)


finetune_layoutlm = define_dagstermill_asset(
    name="layoutlm",
    notebook_path=file_relative_path(__file__, "notebooks/layoutlm.ipynb"),
    save_notebook_on_failure=True,
    config_schema={
        "ex21_training_data": Field(
            String,
            default_value="v0.2",
            is_required=False,
            description="Folder on GCS to pull training data from.",
        ),
    },
)
finetune_layoutlm_job = define_asset_job(
    "finetune_layoutlm",
    description="Execute notebook to finetune layoutlm and log to mlflow.",
    selection=[finetune_layoutlm],
    executor_def=in_process_executor,
)


exhibit21_extraction_validation = define_dagstermill_asset(
    name="validate_exhibit21_extractor",
    notebook_path=file_relative_path(
        __file__, "notebooks/exhibit21_inference_model.ipynb"
    ),
    ins={
        "ex21_validation_set": AssetIn(),
        "ex21_validation_inference_dataset": AssetIn(),
    },
    save_notebook_on_failure=True,
    config_schema={
        "layoutlm_model_uri": Field(
            String,
            default_value="runs:/c81cd8c91900476a8362c54fa3a020fc/layoutlm_extractor",
            is_required=False,
            description="Folder on GCS to pull training data from.",
        ),
    },
)
exhibit21_extraction_validation_job = define_asset_job(
    "exhibit21_extraction_validation",
    description="Execute full Exhibit 21 extraction and validate with labeled validation data.",
    selection=[exhibit21_extraction_validation]
    + ex_21.data.ex21_extraction_validation_assets,
    executor_def=in_process_executor,
)


train_exhibit21_layout_classifier = define_dagstermill_asset(
    name="exhibit21_layout_classifier_model",
    notebook_path=file_relative_path(
        __file__, "notebooks/exhibit21_layout_classifier.ipynb"
    ),
    ins={
        "ex21_layout_labels": AssetIn(),
        "ex21_layout_classifier_training_dataset": AssetIn(),
    },
    save_notebook_on_failure=True,
)
exhibit21_layout_classifier_training_job = define_asset_job(
    "train_exhibit21_layout_classifier",
    description="Train exhibit 21 layout classifier used to filter poorly formatted filings.",
    selection=[train_exhibit21_layout_classifier]
    + ex_21.data.ex21_layout_classifier_assets,
    executor_def=in_process_executor,
)


defs = Definitions(
    assets=basic_10k_assets
    + ex21_assets
    + shared_assets
    + [
        exhibit21_extraction_validation,
        finetune_layoutlm,
        train_exhibit21_layout_classifier,
    ]
    + ex21_data_assets,
    jobs=[
        basic_10k_production_job,
        basic_10k_validation_job,
        exhibit21_production_job,
        finetune_layoutlm_job,
        exhibit21_extraction_validation_job,
        exhibit21_layout_classifier_training_job,
    ],
    resources={
        "cloud_interface": cloud_interface_resource,
        "mlflow_interface": mlflow_interface_resource,
        "pandas_parquet_io_manager": PandasParquetIOManager(
            base_path=UPath("gs://sec10k-outputs/v2")
        ),
        "pickle_gcs_io_manager": PickleUPathIOManager(
            base_path=UPath("gs://sec10k-outputs/dagster_storage")
        ),
        "pyfunc_model_io_manager": MlflowPyfuncModelIOManager(
            mlflow_interface=mlflow_interface_resource
        ),
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
    }
    | mlflow_train_test_io_managers,
)
