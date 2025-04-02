"""Implement record linkage model between SEC companies and EIA utilities."""

from dagster import (
    AssetKey,
    AssetSpec,
    Definitions,
    StaticPartitionsDefinition,
    load_assets_from_modules,
)
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

from ..sec10k.extract import year_quarter_partitions
from . import transform_eia_input, transform_sec_input

eia_assets = load_assets_from_modules([transform_eia_input])
sec_assets = load_assets_from_modules([transform_sec_input])

record_linkage_job = model_jobs.create_production_model_job(
    "sec_eia_record_linkage",
    transform_eia_input.production_assets + transform_sec_input.production_assets,
)

# Create year_quarter partitions
completed_partitions = StaticPartitionsDefinition(
    [
        year_quarter
        for year_quarter in year_quarter_partitions.get_partition_keys()
        if year_quarter
        not in ["2018q1", "2018q2", "2019q1", "2020q1", "2021q1", "2022q1"]
    ]
)

basic_10k_company_info = AssetSpec(
    key=AssetKey("basic_10k_company_info"), partitions_def=completed_partitions
).with_io_manager_key("pandas_parquet_io_manager")

ex21_company_ownership_info = AssetSpec(
    key=AssetKey("ex21_company_ownership_info"), partitions_def=completed_partitions
).with_io_manager_key("pandas_parquet_io_manager")

sec10k_filing_metadata = AssetSpec(
    key=AssetKey("sec10k_filing_metadata"), partitions_def=completed_partitions
).with_io_manager_key("io_manager")

defs = Definitions(
    sec_assets
    + eia_assets
    + [basic_10k_company_info, ex21_company_ownership_info, sec10k_filing_metadata],
    jobs=[record_linkage_job],
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
