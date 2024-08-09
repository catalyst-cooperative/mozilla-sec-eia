"""Implement top level extraction methods and tooling."""

import io
import logging
from importlib import resources

import mlflow
import pandas as pd
import pandera as pa
from mlflow.entities import Run

from mozilla_sec_eia import basic_10k
from mozilla_sec_eia.utils.cloud import GCSArchive, initialize_mlflow

logger = logging.getLogger(f"catalystcoop.{__name__}")


class ExtractionMetadataSchema(pa.DataFrameModel):
    """Define the required schema for extraction metadata.

    Extra columns are permitted, but these are required for computing extraction metrics.
    """

    filename: pa.typing.Index[str] = pa.Field(check_name=True)
    success: bool = pa.Field(coerce=True)


def _load_artifact_as_csv(run: Run, artifact_name: str) -> pd.DataFrame:
    """Download a CSV and parse to DataFrame from mlflow tracking server."""
    df = pd.read_csv(
        io.StringIO(mlflow.artifacts.load_text(run.info.artifact_uri + artifact_name))
    )
    return df


def _log_artifact_as_csv(
    artifact: pd.DataFrame, artifact_name: str, index: bool = True
):
    """Upload a DataFrame as a CSV to mlflow tracking server."""
    return mlflow.log_text(artifact.to_csv(index=index), artifact_name)


def _get_most_recent_run(experiment_name: str):
    """Search mlflow for most recent extraction run with specified experiment name."""
    run_metadata = mlflow.search_runs(experiment_names=[experiment_name])

    # Mlflow returns runs ordered by their runtime, so it's easy to grab the latest run
    # This assert will ensure this doesn't silently break if the ordering changes
    assert run_metadata.loc[0, "end_time"] == run_metadata["end_time"].max()
    return mlflow.get_run(run_metadata.loc[0, "run_id"])


def _get_filings_to_extract(
    experiment_name: str,
    metadata: pd.DataFrame,
    continue_run: bool = False,
    num_filings: int = -1,
):
    """Get filings that should be extracted by run.

    Args:
        experiment_name: Name of mlflow experiment.
        metadata: Metadata for full set of filings to potentially extract.
        continue_run: Whether to continue a previous extraction run.
        num_filings: Number of filings to extract.
    """
    extraction_metadata = pd.DataFrame(
        {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
    ).set_index("filename")
    extracted = pd.DataFrame()
    run_id = None
    if continue_run:
        most_recent_run = _get_most_recent_run(experiment_name)
        extraction_metadata = ExtractionMetadataSchema.validate(
            _load_artifact_as_csv(
                most_recent_run, "/extraction_metadata.csv"
            ).set_index("filename")
        )
        extracted = _load_artifact_as_csv(most_recent_run, "/extracted.csv")
        run_id = most_recent_run.info.run_id

    filings_to_extract = metadata[~metadata["filename"].isin(extraction_metadata.index)]
    if num_filings > 0:
        filings_to_extract = filings_to_extract.sample(num_filings)
    return (
        filings_to_extract,
        extraction_metadata,
        extracted,
        run_id,
    )


def _get_experiment_name(dataset: str, experiment_suffix: str | None = None) -> str:
    experiment_name = f"{dataset}_extraction"
    if experiment_suffix is not None:
        experiment_name += f"_{experiment_suffix}"
    return experiment_name


def compute_validation_metrics(
    computed_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    value_col: str,
) -> dict:
    """Compute precision and recall for extraction compared to validation set.

    Arg:
        computed_set: Extracted data.
        validation_set: Expected extraction results.
        value_col: Column to compare when computing metrics.
    """
    # Get initial length of both sets
    computed_len = len(computed_set)
    validation_len = len(validation_set)

    # Get index of rows only in one set and make Null in other set
    idx_validation_only = validation_set.index.difference(computed_set.index)
    padded_compute_set = pd.concat(
        [
            computed_set[value_col],
            pd.Series([None] * len(idx_validation_only), index=idx_validation_only),
        ]
    ).sort_index()
    idx_compute_only = computed_set.index.difference(validation_set.index)
    padded_validation_set = pd.concat(
        [
            validation_set[value_col],
            pd.Series([None] * len(idx_compute_only), index=idx_compute_only),
        ]
    ).sort_index()

    true_positives = (padded_compute_set == padded_validation_set).sum()

    return {
        "precision": true_positives / computed_len,
        "recall": true_positives / validation_len,
    }


def validate_extraction(dataset: str):
    """Run extraction on validation set and compare results to labeled data."""
    validation_set = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data") / f"{dataset}_labels.csv"
    )

    # Get metadata for labelled filings
    archive = GCSArchive()
    to_extract = archive.get_metadata(filenames=list(validation_set["filename"]))

    # Extract data from filings
    extracted = extract_filings(
        dataset=dataset, metadata=to_extract, experiment_suffix="validation"
    )
    # Set index for validation set based on returned extracted DF
    validation_set = validation_set.set_index(extracted.index.names)

    # Get extraction run from mlflow and start again to log validation metrics
    experiment_name = _get_experiment_name(dataset, experiment_suffix="validation")
    run = _get_most_recent_run(experiment_name)
    with mlflow.start_run(run_id=run.info.run_id):
        # Compute metrics and log
        if dataset == "basic_10k":
            mlflow.log_metrics(
                compute_validation_metrics(extracted, validation_set, "value")
            )
        # Log validation set used to compute metrics
        _log_artifact_as_csv(validation_set, "labels.csv")


def extract_filings(
    dataset: str,
    continue_run: bool = False,
    num_filings: int = -1,
    metadata: pd.DataFrame | None = None,
    experiment_suffix: str | None = None,
) -> pd.DataFrame:
    """Extra data from SEC 10k and exhibit 21 filings.

    This function takes several parameters to decide which filings to extract data
    from. If `continue_run` is set, it will search the mlflow tracking server for
    the most recent extraction run for the specified dataset, and download corresponding
    metadata and extraction results. It will then filter out any filings that were
    already extracted in the run. This is useful for testing to be able perform extraction
    on subsets of the data and continue where it left off. If `metadata` is passed
    in, this is expected to be a selection of filing metadata that specifies exactly
    which filings to extract. This is used for validation to only extract filings in
    the validation set.

    Args:
        dataset: Data to extract, should be 'basic_10k' or 'ex21'.
        continue_run: Whether to continue a previous extraction run.
        num_filings: Number of filings to extract in run.
        metadata: Specific selection of filing metadata to extract.
        experiment_suffix: Add to mlflow run to differentiate run from basic extraction.
    """
    if dataset not in ["ex21", "basic_10k"]:
        raise RuntimeError(
            f"{dataset} is not a valid dataset. Must be 'ex21' or 'basic_10k'."
        )

    initialize_mlflow()

    # Get filing metadata if not passed in explicitly
    archive = GCSArchive()
    if metadata is None:
        metadata = archive.get_metadata()

    experiment_name = _get_experiment_name(dataset, experiment_suffix=experiment_suffix)

    # Get filings to extract as well as any existing metadata for run
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
        # Extract data for desired filings
        if dataset == "basic_10k":
            extraction_metadata, extracted = basic_10k.extract(
                filings_to_extract,
                extraction_metadata,
                extracted,
                archive,
            )
        else:
            # TODO: add in ex.21 extraction here
            # model_checkpoint = load_model()
            # model = model_checkpoint["model"]
            # processor = model_checkpoint["tokenizer"]
            # need to create a PDFs dir or can just give filings
            # extraction_metadata, extracted = perform_inference()
            logger.warning("Exhibit 21 extraction is not yet implemented.")

        # Use metadata to log generic metrics
        extraction_metadata = ExtractionMetadataSchema.validate(extraction_metadata)
        mlflow.log_metrics(
            {
                "num_failed": (~extraction_metadata["success"]).sum(),
                "ratio_extracted": len(extraction_metadata) / len(metadata),
            }
        )

        # Log the extraction results + metadata for future reference/analysis
        _log_artifact_as_csv(extraction_metadata, "extraction_metadata.csv")
        _log_artifact_as_csv(extracted, "extracted.csv")
    logger.info(
        f"Finished extracting {len(extraction_metadata)} filings from {dataset}."
    )
    return extracted
