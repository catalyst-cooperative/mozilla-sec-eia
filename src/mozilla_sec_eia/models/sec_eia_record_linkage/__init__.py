"""Implement record linkage model between SEC companies and EIA utilities."""

from dagster import Definitions, load_assets_from_modules
from dagstermill import (
    ConfigurableLocalOutputNotebookIOManager,
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
from mozilla_sec_eia.models.sec10k.utils.cloud import cloud_interface_resource

from . import transform_eia_input, transform_sec_input

eia_assets = load_assets_from_modules([transform_eia_input])
sec_assets = load_assets_from_modules([transform_sec_input])

eia_input_table_production_job = model_jobs.create_production_model_job(
    "eia_input_table_creation", transform_eia_input.production_assets
)
sec_input_table_production_job = model_jobs.create_production_model_job(
    "sec_input_table_creation", transform_sec_input.production_assets
)

defs = Definitions(
    sec_assets,
    jobs=[eia_input_table_production_job, sec_input_table_production_job],
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
