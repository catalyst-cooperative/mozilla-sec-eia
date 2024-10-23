"""Implement tooling to interface with mlflow experiment tracking."""

from dagster import Config, asset
from pydantic import create_model

from .mlflow_io_managers import (
    MlflowBaseIOManager,
    MlflowMetricsIOManager,
    MlflowPandasArtifactIOManager,
    MlflowPyfuncModelIOManager,
)
from .mlflow_resource import (
    MlflowInterface,
    configure_mlflow,
    get_most_recent_run,
)


def pyfunc_model_asset_factory(name: str, mlflow_run_uri: str):
    """Create asset for loading a model logged to mlflow."""
    PyfuncConfig = create_model(  # NOQA: N806
        f"PyfuncConfig{name}", mlflow_run_uri=(str, mlflow_run_uri), __base__=Config
    )

    @asset(
        name=name,
        io_manager_key="pyfunc_model_io_manager",
    )
    def _model_asset(config: PyfuncConfig):
        return config.mlflow_run_uri

    return _model_asset


def get_mlflow_io_manager(
    key: str,
    mlflow_interface: MlflowInterface | None = None,
    pandas_file_type: str = "parquet",
) -> MlflowBaseIOManager:
    """Construct IO-manager based on key."""
    if key == "mlflow_pandas_artifact_io_manager":
        io_manager = MlflowPandasArtifactIOManager(
            file_type=pandas_file_type,
            mlflow_interface=mlflow_interface,
        )
    elif key == "mlflow_metrics_io_manager":
        io_manager = MlflowMetricsIOManager(
            mlflow_interface=mlflow_interface,
        )
    else:
        raise RuntimeError(f"MlFlow IO-manager, {key}, does not exist.")

    return io_manager


mlflow_interface_resource = MlflowInterface.configure_at_launch()
mlflow_train_test_io_managers = {
    "mlflow_metrics_io_manager": get_mlflow_io_manager(
        "mlflow_metrics_io_manager",
        mlflow_interface=mlflow_interface_resource,
    ),
    "mlflow_pandas_artifact_io_manager": get_mlflow_io_manager(
        "mlflow_pandas_artifact_io_manager",
        mlflow_interface=mlflow_interface_resource,
    ),
}
