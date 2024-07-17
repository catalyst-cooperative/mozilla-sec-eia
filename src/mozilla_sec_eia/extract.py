"""Implement top level extraction method and tooling."""

import io
import logging

import mlflow
import pandas as pd
from mlflow.entities import Run

from mozilla_sec_eia import basic_10k
from mozilla_sec_eia.utils.cloud import GCSArchive, initialize_mlflow

logger = logging.getLogger(f"catalystcoop.{__name__}")


def _load_artifact_as_csv(run: Run, artifact_name: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(mlflow.artifacts.load_text(run.info.artifact_uri + artifact_name))
    )


def _log_artifact_as_csv(
    artifact: pd.DataFrame, artifact_name: str, index: bool = True
):
    return mlflow.log_text(artifact.to_csv(index=index), artifact_name)


def _get_most_recent_run(experiment_name: str):
    """Search mlflow for most recent basic 10k extraction run."""
    run_metadata = mlflow.search_runs(experiment_names=[experiment_name])
    return mlflow.get_run(
        run_metadata[run_metadata["end_time"] == run_metadata["end_time"].max()][
            "run_id"
        ][0]
    )


def _get_filings_to_extract(
    experiment_name: str,
    metadata: pd.DataFrame,
    continue_run: bool = False,
    num_filings: int = -1,
):
    """Get filings that should be extracted by run."""
    extraction_metadata = pd.DataFrame(
        {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
    )
    extracted = pd.DataFrame()
    run_id = None
    if continue_run:
        most_recent_run = _get_most_recent_run(experiment_name)
        extraction_metadata = _load_artifact_as_csv(
            most_recent_run, "/extraction_metadata.csv"
        )
        extracted = _load_artifact_as_csv(most_recent_run, "/extracted.csv")
        run_id = most_recent_run.info.run_id

    filings_to_extract = metadata[
        ~metadata["filename"].isin(extraction_metadata["filename"])
    ]
    if num_filings > 0:
        filings_to_extract = filings_to_extract.sample(num_filings)
    return (
        filings_to_extract,
        extraction_metadata.set_index("filename"),
        extracted,
        run_id,
    )


def extract_filings(
    extractors: list[str], continue_run: bool = False, num_filings: int = -1
):
    """Extra data from SEC 10k and exhibit 21 filings."""
    if any(bad_args := [arg for arg in extractors if arg not in ["ex21", "basic_10k"]]):
        raise RuntimeError(f"The following extractors do not exist: {bad_args}")

    initialize_mlflow()
    archive = GCSArchive()
    metadata = archive.get_metadata()

    for dataset in extractors:
        experiment_name = f"{dataset}_extraction"
        filings_to_extract, extraction_metadata, extracted, run_id = (
            _get_filings_to_extract(
                experiment_name,
                metadata,
                continue_run=continue_run,
                num_filings=num_filings,
            )
        )
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id):
            if dataset == "basic_10k":
                extraction_metadata, extracted = basic_10k.extract(
                    filings_to_extract,
                    extraction_metadata,
                    extracted,
                    archive,
                )
            mlflow.log_metrics(
                {
                    "num_failed": (~extraction_metadata["success"]).sum(),
                    "ratio_extracted": len(extraction_metadata) / len(metadata),
                }
            )
            _log_artifact_as_csv(extraction_metadata, "extraction_metadata.csv")
            _log_artifact_as_csv(extracted, "extracted.csv", index=False)
