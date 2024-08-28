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

from .mlflow_resource import ExperimentTracker

logger = logging.getLogger(f"catalystcoop.{__name__}")


class MlflowPandasArtifactIOManager(ConfigurableIOManager):
    """Implement IO manager for logging/loading parquet files as mlflow artifacts."""

    experiment_tracker: ExperimentTracker
    #: By default handles artifacts from current run, but can be used with previous run.
    use_previous_mlflow_run: bool = False
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
        if self.use_previous_mlflow_run:
            raise NotImplementedError(
                "MlflowPandasArtifactIOManager can not be used to add artifacts to completed run."
            )

        if self.file_type == "csv":
            self._log_artifact_as_csv(df, artifact_name=f"{context.name}.csv")
        else:
            self._log_artifact_as_parquet(df, artifact_name=f"{context.name}.parquet")

    def _get_run_info(self) -> Run:
        """Use `dagster_run_id` and `use_previous_mlflow_run` to get run info from appropriate mlflow run."""
        dagster_run_id = self.experiment_tracker.get_run_id()
        filter_string = f"tags.dagster_run_id='{dagster_run_id}'"
        if self.use_previous_mlflow_run:
            filter_string = f"tags.dagster_run_id!='{dagster_run_id}'"

        run_metadata = mlflow.search_runs(
            experiment_names=[self.experiment_tracker.experiment_name],
            filter_string=filter_string,
        )

        # Mlflow returns runs ordered by their runtime, so it's easy to grab the latest run
        return mlflow.get_run(run_metadata.loc[0, "run_id"])

    def load_input(self, context: InputContext) -> pd.DataFrame:
        """Handle loading dataframes from mlflow run artifacts."""
        mlflow_run = self._get_run_info()

        if self.file_type == "csv":
            df = self._load_artifact_as_csv(
                mlflow_run, artifact_name=f"{context.name}.csv"
            )
        else:
            df = self._load_artifact_as_parquet(
                mlflow_run, artifact_name=f"{context.name}.parquet"
            )

        return df
