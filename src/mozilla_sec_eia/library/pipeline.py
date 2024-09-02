"""Implement helper methods for constructing dagster jobs.

Methods defined here are the main interface for constructing PUDL model jobs.
`create_production_pipeline` will produce a dagster job that will use the default
multi-process executor to run a PUDL model. `create_validation_pipeline` is meant for
testing/validating models with an mlflow run backing the dagster run for logging.
"""

import mlflow
from dagster import (
    AssetsDefinition,
    HookContext,
    ResourceDefinition,
    define_asset_job,
    failure_hook,
    in_process_executor,
    success_hook,
)
from mlflow.entities import RunStatus

PUDL_PIPELINE_PRODUCTION_JOBS = []
PUDL_PIPELINE_PRODUCTION_ASSETS = []
PUDL_PIPELINE_PRODUCTION_RESOURCES = {}

PUDL_PIPELINE_VALIDATION_JOBS = []
PUDL_PIPELINE_VALIDATION_ASSETS = []
PUDL_PIPELINE_VALIDATION_RESOURCES = {}


def create_production_pipeline(
    pipeline_name: str,
    assets: list[AssetsDefinition],
    resources: dict[str, ResourceDefinition],
    **kwargs,
):
    """Construct a dagster job and supply Definitions with assets and resources."""
    PUDL_PIPELINE_PRODUCTION_JOBS.append(
        define_asset_job(
            pipeline_name,
            selection=assets,
            **kwargs,
        )
    )
    PUDL_PIPELINE_PRODUCTION_ASSETS.extend(assets)
    PUDL_PIPELINE_PRODUCTION_RESOURCES.update(resources)


@success_hook(required_resource_keys={"mlflow_interface"})
def log_op_config(context: HookContext):
    """Log any config supplied to ops/assets in validation job to mlflow tracking server."""
    if context.op_config is not None:
        mlflow.log_params(context.op_config)


@failure_hook(required_resource_keys={"mlflow_interface"})
def end_run_on_failure(context: HookContext):
    """Inform mlflow about job failure."""
    if isinstance(context.op_exception, KeyboardInterrupt):
        mlflow.end_run(status=RunStatus.to_string(RunStatus.KILLED))
    else:
        mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))


def create_validation_pipeline(
    pipeline_name: str,
    assets: list[AssetsDefinition],
    resources: dict[str, ResourceDefinition],
    **kwargs,
):
    """Construct a dagster job and supply Definitions with assets and resources."""
    PUDL_PIPELINE_VALIDATION_JOBS.append(
        define_asset_job(
            pipeline_name,
            selection=assets,
            executor_def=in_process_executor,
            hooks={log_op_config, end_run_on_failure},
            # Configure mlflow_interface for job with appropriate experiment name
            config={
                "ops": {},
                "resources": {
                    "mlflow_interface": {
                        "config": {
                            "experiment_name": pipeline_name,
                            "tracking_enabled": True,
                        }
                    }
                },
            },
            **kwargs,
        )
    )
    PUDL_PIPELINE_VALIDATION_ASSETS.extend(assets)
    PUDL_PIPELINE_VALIDATION_RESOURCES.update(resources)
