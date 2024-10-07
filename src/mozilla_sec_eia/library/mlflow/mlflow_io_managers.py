"""Implement IO managers for loading models/artifacts from tracking server."""

import io
import logging
import tempfile
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext
from mlflow.entities import Run

from .mlflow_resource import MlflowInterface

logger = logging.getLogger(f"catalystcoop.{__name__}")


class MlflowBaseIOManager(ConfigurableIOManager):
    """Specify base config and implement helper functions for mlflow io-managers."""

    mlflow_interface: MlflowInterface

    def _get_run_info(self) -> Run:
        """Get mlflow `Run` object using current run id."""
        return mlflow.get_run(self.mlflow_interface.mlflow_run_id)


class MlflowPyfuncModelIOManager(MlflowBaseIOManager):
    """IO Manager to load pyfunc models from tracking server."""

    uri: str | None = None

    def handle_output(self, context, obj):
        """Outputs not implemented."""
        raise NotImplementedError("Logging models not supported by io manager.")

    def load_input(self, context: InputContext):
        """Load pyfunc model with mlflow server."""
        cache_path = (
            self.mlflow_interface.dagster_home_path / "model_cache" / context.name
        )
        cache_path.mkdir(exist_ok=True, parents=True)

        model_uri = self.uri
        if model_uri is None:
            model_uri = f"models:/{context.name}"

        return mlflow.pyfunc.load_model(
            model_uri,
            dst_path=cache_path,
        )


class MlflowPandasArtifactIOManager(MlflowBaseIOManager):
    """Implement IO manager for logging/loading dataframes as mlflow artifacts."""

    file_type: Literal["parquet", "csv"] = "parquet"

    def _load_artifact_as_csv(self, run: Run, artifact_name: str) -> pd.DataFrame:
        """Download a CSV and parse to DataFrame from mlflow tracking server."""
        df = pd.read_csv(
            io.StringIO(
                mlflow.artifacts.load_text(run.info.artifact_uri + f"/{artifact_name}")
            )
        )
        return df

    def _log_artifact_as_csv(
        self, artifact: pd.DataFrame, artifact_name: str, index: bool = True
    ):
        """Upload a DataFrame as a CSV to mlflow tracking server."""
        return mlflow.log_text(artifact.to_csv(index=index), artifact_name)

    def _load_artifact_as_parquet(self, run: Run, artifact_name: str) -> pd.DataFrame:
        """Download a CSV and parse to DataFrame from mlflow tracking server."""
        df = pd.read_parquet(run.info.artifact_uri + f"/{artifact_name}")
        return df

    def _log_artifact_as_parquet(
        self, artifact: pd.DataFrame, artifact_name: str, index: bool = True
    ):
        """Upload a DataFrame as a CSV to mlflow tracking server."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / artifact_name
            artifact.to_parquet(parquet_path, index=index)
            return mlflow.log_artifact(parquet_path, artifact_name)

    def _get_dagster_run_id(self, context: InputContext | OutputContext) -> str:
        """Return dagster run id of current dagster run."""
        return context.get_identifier()[0]

    def handle_output(self, context: OutputContext, df: pd.DataFrame):
        """Attach dataframe to run as artifact."""
        if self.file_type == "csv":
            self._log_artifact_as_csv(df, artifact_name=f"{context.name}.csv")
        else:
            self._log_artifact_as_parquet(df, artifact_name=f"{context.name}.parquet")

    def load_input(self, context: InputContext) -> pd.DataFrame:
        """Handle loading dataframes from mlflow run artifacts."""
        mlflow_run = self._get_run_info()

        if self.file_type == "csv":
            df = self._load_artifact_as_csv(
                mlflow_run, artifact_name=f"{context.upstream_output.name}.csv"
            )
        else:
            df = self._load_artifact_as_parquet(
                mlflow_run, artifact_name=f"{context.upstream_output.name}.parquet"
            )

        return df


class MlflowMetricsIOManager(MlflowBaseIOManager):
    """Log/load metrics from mlflow tracking server."""

    def handle_output(self, context: OutputContext, obj: dict[str, float]):
        """Load metrics to mlflow run/experiment created by `MlflowInterface`."""
        mlflow.log_metrics(obj)

    def load_input(self, context: InputContext) -> dict[str, float]:
        """Log metrics to mlflow run/experiment created by `MlflowInterface`."""
        run = self._get_run_info()
        return run.data.metrics
