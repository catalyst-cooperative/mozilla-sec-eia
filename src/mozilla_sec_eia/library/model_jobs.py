"""Implement helper methods for constructing dagster jobs.

Methods defined here are the main interface for constructing PUDL model jobs.
`create_production_model_job` will produce a dagster job that will use the default
multi-process executor to run a PUDL model. `create_validation_model_job` is meant for
testing/validating models with an mlflow run backing the dagster run for logging.
To avoid problems with mlflow runs, test/validation jobs are run with the dagster
in process executor.
"""

import mlflow
from dagster import (
    AssetsDefinition,
    HookContext,
    JobDefinition,
    define_asset_job,
    failure_hook,
    in_process_executor,
    success_hook,
)
from mlflow.entities import RunStatus


def create_production_model_job(
    job_name: str,
    assets: list[AssetsDefinition],
    concurrency_limit: int | None = None,
    tag_concurrency_limits: list[dict] | None = None,
    **kwargs,
) -> JobDefinition:
    """Construct a dagster job and supply Definitions with assets and resources."""
    config = {
        "ops": {},
        "resources": {
            "mlflow_interface": {
                "config": {
                    "experiment_name": job_name,
                    "tracking_enabled": False,
                }
            }
        },
    }
    if (concurrency_limit is not None) or (tag_concurrency_limits is not None):
        config["execution"] = {"config": {"multiprocess": {}}}
        if concurrency_limit is not None:
            config["execution"]["config"]["multiprocess"][
                "max_concurrent"
            ] = concurrency_limit
        else:
            config["execution"]["config"]["multiprocess"][
                "tag_concurrency_limits"
            ] = tag_concurrency_limits

    return define_asset_job(
        job_name,
        selection=assets,
        config=config,
        **kwargs,
    )


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


def create_validation_model_job(
    job_name: str,
    assets: list[AssetsDefinition],
    **kwargs,
):
    """Construct a dagster job and supply Definitions with assets and resources."""
    return define_asset_job(
        job_name,
        selection=assets,
        executor_def=in_process_executor,
        hooks={log_op_config, end_run_on_failure},
        # Configure mlflow_interface for job with appropriate experiment name
        config={
            "ops": {},
            "resources": {
                "mlflow_interface": {
                    "config": {
                        "experiment_name": job_name,
                        "tracking_enabled": True,
                    }
                }
            },
        },
        **kwargs,
    )


def create_training_job(
    job_name: str,
    assets: list[AssetsDefinition],
    **kwargs,
):
    """Construct a dagster job meant to train a model and log with mlflow."""
    # For now training job config is the same as validation
    return create_validation_model_job(job_name, assets, **kwargs)
