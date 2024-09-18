"""This module implements experiment tracking tooling using mlflow as a backend.

:class:`ExperimentTracker`'s are created using an op factory :func:`experiment_tracker_factory`
and can be passed around to op's which make up a PUDL model. This class will maintain
state between ops, ensuring that all parameters and metrics are logged to the appropriate
mlflow run. The following command will launch the mlflow UI to view model results:
`mlflow ui --backend-store-uri {tracking_uri}`. `tracking_uri` by default will point
to a file named 'experiments.sqlite' in the base directory of your PUDL repo, but
this is a configurable value, which can be found in the dagster UI.
"""

import logging
import os
from contextlib import contextmanager

import mlflow
from dagster import ConfigurableResource, EnvVar, InitResourceContext
from google.cloud import secretmanager
from pydantic import PrivateAttr

logger = logging.getLogger(f"catalystcoop.{__name__}")


def _configure_mlflow(tracking_uri: str, project: str):
    """Do runtime configuration of mlflow."""
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = _get_tracking_password(
        tracking_uri, project
    )
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE"] = "20971520"
    os.environ["MLFLOW_GCS_UPLOAD_CHUNK_SIZE"] = "20971520"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "900"
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


def _get_tracking_password(tracking_uri: str, project: str, version_id: str = "latest"):
    """Get tracking server password from gcloud secrets."""
    # Password not required for local use
    if "sqlite" not in tracking_uri:
        # Create the Secret Manager client.
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version.
        name = f"projects/{project}/secrets/mlflow_admin_password/versions/{version_id}"

        # Access the secret version.
        response = client.access_secret_version(name=name)

        # Return the decoded payload.
        return response.payload.data.decode("UTF-8")
    return ""


class MlflowInterface(ConfigurableResource):
    """Dagster resource to interface with mlflow tracking server.

    This resource handles configuring mlflow to interface with a remote tracking server.
    When `tracking_enabled` is set to True, this resource will also start an mlflow
    run that can be used to log metrics/paramaters/artifcats which will be associated
    with a validation or training run. In most cases this resource does not need to
    be referenced directly, and instead the io-mangers defined in
    :mod:`.mlflow_io_managers` should be used.

    Note: `tracking_enabled` SHOULD NOT be set when using a dagster multi-process
    executor. mlflow will create a new run for every process, which gets very messy.
    """

    tracking_uri: str = EnvVar("MLFLOW_TRACKING_URI")
    tracking_enabled: bool = True
    artifact_location: str | None = None
    experiment_name: str
    tags: dict = {}
    project: str = EnvVar("GCS_PROJECT")

    _mlflow_run_id: str = PrivateAttr()

    @contextmanager
    def yield_for_execution(
        self,
        context: InitResourceContext,
    ) -> "MlflowInterface":
        """Create experiment tracker for specified experiment."""
        dagster_run_id = context.run_id
        self._mlflow_run_id = None
        _configure_mlflow(self.tracking_uri, self.project)

        if self.tracking_enabled:
            # Get run_id associated with current dagster run
            experiment_id = self.get_or_create_experiment(
                experiment_name=self.experiment_name,
                artifact_location=self.artifact_location,
            )
            # Create new run under specified experiment
            with mlflow.start_run(
                experiment_id=experiment_id,
                tags=self.tags | {"dagster_run_id": dagster_run_id},
            ) as run:
                self._mlflow_run_id = run.info.run_id
                yield self
        else:
            yield self

    @property
    def mlflow_run_id(self) -> str | None:
        """Return run id of current run."""
        return self._mlflow_run_id

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
