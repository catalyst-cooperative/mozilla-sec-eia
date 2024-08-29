"""Implement tooling to interface with mlflow experiment tracking."""

from .mlflow_io_managers import (
    MlflowBaseIOManager,
    MlflowMetricsIOManager,
    MlflowPandasArtifactIOManager,
)
from .mlflow_resource import (
    ExperimentTracker,
    experiment_tracker_teardown_factory,
    get_most_recent_run,
)


def get_mlflow_io_manager(
    key: str, experiment_tracker: ExperimentTracker, pandas_file_type: str = "parquet"
) -> MlflowBaseIOManager:
    """Construct IO-manager based on key."""
    if key == "mlflow_pandas_artifact_io_manager":
        io_manager = MlflowPandasArtifactIOManager(
            file_type=pandas_file_type,
            experiment_tracker=experiment_tracker,
        )
    elif key == "previous_run_mlflow_pandas_artifact_io_manager":
        io_manager = MlflowPandasArtifactIOManager(
            file_type=pandas_file_type,
            experiment_tracker=experiment_tracker,
            use_previous_mlflow_run=True,
        )
    elif key == "mlflow_metrics_io_manager":
        io_manager = MlflowMetricsIOManager(
            experiment_tracker=experiment_tracker,
        )
    else:
        raise RuntimeError(f"MlFlow IO-manager, {key}, does not exist.")

    return io_manager
