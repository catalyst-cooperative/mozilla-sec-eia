"""Implement tooling to interface with mlflow experiment tracking."""

from .mlflow_io_managers import (
    MlflowBaseIOManager,
    MlflowMetricsIOManager,
    MlflowPandasArtifactIOManager,
)
from .mlflow_resource import (
    MlflowInterface,
    configure_mlflow,
    get_most_recent_run,
)


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
