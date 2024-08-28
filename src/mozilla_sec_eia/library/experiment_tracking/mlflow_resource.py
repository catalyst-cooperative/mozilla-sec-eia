"""This module implements experiment tracking tooling using mlflow as a backend.

:class:`ExperimentTracker`'s are created using an op factory :func:`experiment_tracker_factory`
and can be passed around to op's which make up a PUDL model. This class will maintain
state between ops, ensuring that all parameters and metrics are logged to the appropriate
mlflow run. The following command will launch the mlflow UI to view model results:
`mlflow ui --backend-store-uri {tracking_uri}`. `tracking_uri` by default will point
to a file named 'experiments.sqlite' in the base directory of your PUDL repo, but
this is a configurable value, which can be found in the dagster UI.
"""

import atexit
import logging
import os
from contextlib import contextmanager

import mlflow
from dagster import ConfigurableResource, InitResourceContext, op
from google.cloud import secretmanager

logger = logging.getLogger(f"catalystcoop.{__name__}")


class ExperimentTracker(ConfigurableResource):
    """Class to manage tracking a machine learning model using MLflow.

    The following command will launch the mlflow UI to view model results:
    `mlflow ui --backend-store-uri {tracking_uri}`. From here, you can compare metrics
    from multiple runs, and track performance.

    This class is designed to be created using the `op` :func:`create_experiment_tracker`.
    This allows the `ExperimentTracker` to be passed around within a Dagster `graph`,
    and be used for mlflow logging in any of the `op`'s that make up the `graph`. This
    is useful because Dagster executes `op`'s in separate processes, while mlflow does
    not maintain state between processes. This design also allows configuration of
    the ExperimentTracker to be set from the Dagster UI.

    Currently, we are only doing experiment tracking in a local context, but if we were
    to setup a tracking server, we could point the `tracking_uri` at this remote server
    without having to modify the models. Experiment tracking can also be done outside
    of the PUDL context. If doing exploratory work in a notebook, you can use mlflow
    directly in a notebook with the same experiment name used here, and mlflow will
    seamlessly integrate the results with those from PUDL runs.
    """

    tracking_uri: str
    tracking_enabled: bool = True
    artifact_location: str | None = None
    experiment_name: str
    tags: dict = {}
    project: str

    @contextmanager
    def yield_for_execution(
        self,
        context: InitResourceContext,
    ) -> "ExperimentTracker":
        """Create experiment tracker for specified experiment."""
        if self.tracking_enabled:
            self._configure_mlflow()

            # Get run_id associated with current dagster run
            experiment_id = self.get_or_create_experiment(
                experiment_name=self.experiment_name,
                artifact_location=self.artifact_location,
            )
            mlflow_run_id = self._get_mlflow_run_id(context.run_id, experiment_id)

            # Hack to stop mlflow from ending run at process barrier
            # This is borrowed from the official dagster mlflow resource found here:
            # https://github.com/dagster-io/dagster/blob/master/python_modules/libraries/dagster-mlflow/dagster_mlflow/resources.py
            atexit.unregister(mlflow.end_run)

            # Create new run under specified experiment
            with mlflow.start_run(
                run_id=mlflow_run_id,
                experiment_id=experiment_id,
                tags=self.tags | {"dagster_run_id": context.run_id},
            ):
                yield self

    def _get_tracking_password(self, version_id: str = "latest"):
        """Get tracking server password from gcloud secrets."""
        # Create the Secret Manager client.
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version.
        name = f"projects/{self.project}/secrets/mlflow_admin_password/versions/{version_id}"

        # Access the secret version.
        response = client.access_secret_version(name=name)

        # Return the decoded payload.
        return response.payload.data.decode("UTF-8")

    def _configure_mlflow(self):
        """Do runtime configuration of mlflow."""
        os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._get_tracking_password()
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE"] = "20971520"
        os.environ["MLFLOW_GCS_UPLOAD_CHUNK_SIZE"] = "20971520"
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "900"
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    def _get_mlflow_run_id(self, dagster_run_id: str, experiment_id: str):
        """Search for existing run tagged with dagster run id or start new run."""
        run_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.dagster_run_id='{dagster_run_id}'",
        )

        run_id = None
        if not run_df.empty:
            run_id = run_df.loc[0, "run_id"]
        return run_id

    @staticmethod
    def get_or_create_experiment(
        experiment_name: str, artifact_location: str = ""
    ) -> str:
        """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

        This function checks if an experiment with the given name exists within MLflow.
        If it does, the function returns its ID. If not, it creates a new experiment
        with the provided name and returns its ID.

        Returns:
            ID of the existing or newly created MLflow experiment.
        """
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )

        return experiment_id


def experiment_tracker_teardown_factory(
    experiment_name: str,
) -> ExperimentTracker:
    """Use config to create an experiment tracker."""
    atexit.unregister(mlflow.end_run)

    @op(
        name=f"{experiment_name}_tracker_teardown",
        required_resource_keys=["experiment_tracker"],
    )
    def teardown_experiment_tracker(_results):
        mlflow.end_run()

    return teardown_experiment_tracker


def get_most_recent_run(
    experiment_name: str, dagster_run_id: str
) -> mlflow.entities.Run:
    """Search mlflow for most recent extraction run with specified experiment name."""
    run_metadata = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.dagster_run_id!='{dagster_run_id}'",
    )

    # Mlflow returns runs ordered by their runtime, so it's easy to grab the latest run
    # This assert will ensure this doesn't silently break if the ordering changes
    assert run_metadata.loc[0, "end_time"] == run_metadata["end_time"].max()
    return mlflow.get_run(run_metadata.loc[0, "run_id"])
