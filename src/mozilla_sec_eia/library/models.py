"""Provides tooling for developing/tracking ml models within PUDL.

The main interface from this module is the :func:`pudl_model` decorator, which
is meant to be applied to a dagster `graph`. This decorator will handle finding all
configuration for a model/passing configuration to dagster, creating an
:class:`ExperimentTracker` for the model, and ultimately will return a `job`
from the model.

There are a few different ways to provide configuration for a PUDL model. First, configuration will come from default values for any dagster `Config`'s which are associated
with `op`'s which make up the model `graph`. For more info on dagster configuration,
see https://docs.dagster.io/concepts/configuration/config-schema. The next way to
provide configuration is through the yaml file: `pudl.package_data.settings.pudl_models.yml`.
Any configuration in this file should be follow dagster's config-schema formatting,
see the `ferc_to_ferc` entry as an example. Configuration provided this way will
override any default values. The final way to provide configuration is through the
dagster UI. To provide configuration this way, click `Open Launchpad` in the UI, and
values can be edited here. This configuration will override both default values and
yaml configuration, but will only be used for a single run.
"""

import importlib
import logging

import mlflow
import yaml
from dagster import (
    EnvVar,
    GraphDefinition,
    HookContext,
    JobDefinition,
    OpDefinition,
    ResourceDefinition,
    RunConfig,
    failure_hook,
    job,
    op,
    success_hook,
)
from mlflow.entities.run_status import RunStatus

from .experiment_tracking import (
    ExperimentTracker,
    MlflowPandasArtifactIOManager,
    experiment_tracker_teardown_factory,
)

logger = logging.getLogger(f"catalystcoop.{__name__}")
MODEL_RESOURCES = {}
PUDL_MODELS = {}


def get_yml_config(experiment_name: str) -> dict:
    """Load model configuration from yaml file."""
    config_file = (
        importlib.resources.files("pudl.package_data.settings") / "pudl_models.yml"
    )
    config = yaml.safe_load(config_file.open("r"))

    if not (model_config := config.get(experiment_name)):
        raise RuntimeError(f"No {experiment_name} entry in {config_file}")

    return {experiment_name: model_config}


def get_default_config(model_graph: GraphDefinition) -> dict:
    """Get default config values for model."""

    def _get_default_from_ops(node: OpDefinition | GraphDefinition):
        config = {}
        if isinstance(node, GraphDefinition):
            config = {
                "ops": {
                    child_node.name: _get_default_from_ops(child_node)
                    for child_node in node.node_defs
                }
            }
        else:
            if node.config_schema.default_provided:
                config = {"config": node.config_schema.default_value}
            else:
                config = {"config": None}

        return config

    config = {model_graph.name: _get_default_from_ops(model_graph)}
    return config


def get_pudl_model_job_name(experiment_name: str) -> str:
    """Return expected pudl model job name based on experiment_name."""
    return f"{experiment_name}_job"


def pudl_model(
    experiment_name: str,
    mlflow_pandas_io_manager_file_type: str = "parquet",
    resources: dict[str, ResourceDefinition] = {},
    config_from_yaml: bool = False,
) -> JobDefinition:
    """Decorator for an ML model that will handle providing configuration to dagster."""

    def _decorator(model_graph: GraphDefinition):
        model_config = get_default_config(model_graph)
        if config_from_yaml:
            model_config |= get_yml_config(model_graph.name)

        # Add resources to resource dict
        experiment_tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri=EnvVar("MLFLOW_TRACKING_URI"),
            project=EnvVar("GCS_PROJECT"),
        )
        model_resources = {
            "experiment_tracker": experiment_tracker,
            "mlflow_pandas_artifact_io_manager": MlflowPandasArtifactIOManager(
                file_type=mlflow_pandas_io_manager_file_type,
                experiment_tracker=experiment_tracker,
            ),
            "previous_run_mlflow_pandas_artifact_io_manager": MlflowPandasArtifactIOManager(
                use_previous_mlflow_run=True,
                file_type=mlflow_pandas_io_manager_file_type,
                experiment_tracker=experiment_tracker,
            ),
        } | resources

        default_config = RunConfig(
            ops=model_config,
        )

        @op
        def _collect_results(model_graph_output, _implicit_dependencies: list):
            return model_graph_output

        @success_hook(required_resource_keys={"experiment_tracker"})
        def _log_config_hook(context: HookContext):
            if (config := context.op_config) is not None:
                mlflow.log_params(
                    {
                        f"{context.op.name}.{param}": value
                        for param, value in config.items()
                    }
                )

        @failure_hook(required_resource_keys={"experiment_tracker"})
        def _end_mlflow_run_with_failure(context: HookContext):
            exception = context.op_exception

            if isinstance(exception, KeyboardInterrupt):
                mlflow.end_run(status=RunStatus.to_string(RunStatus.KILLED))
            else:
                mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))

        @job(
            name=get_pudl_model_job_name(experiment_name),
            config=default_config,
            hooks={_log_config_hook, _end_mlflow_run_with_failure},
            resource_defs=model_resources,
        )
        def model_job(**kwargs):
            tracker_teardown = experiment_tracker_teardown_factory(
                experiment_name=model_graph.name,
            )
            graph_output = model_graph(**kwargs)

            # Pass output to teardown to create a dependency
            tracker_teardown(graph_output)

        PUDL_MODELS[get_pudl_model_job_name(experiment_name)] = model_job
        return model_job

    return _decorator
