"""Implement top level extraction methods and tooling."""

import io
import logging
import tempfile
from importlib import resources
from pathlib import Path

import mlflow
import pandas as pd
import pandera as pa
from dagster import ConfigurableResource, asset
from mlflow.entities import Run

from mozilla_sec_eia import basic_10k
from mozilla_sec_eia.ex_21.inference import clean_extracted_df, perform_inference
from mozilla_sec_eia.utils.cloud import (
    GCSArchive,
    get_metadata_filename,
)
from mozilla_sec_eia.utils.layoutlm import (
    load_model,
)

logger = logging.getLogger(f"catalystcoop.{__name__}")

DATASETS = ["ex21", "basic_10k"]


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


def _load_artifact_as_parquet(run: Run, artifact_name: str) -> pd.DataFrame:
    """Download a CSV and parse to DataFrame from mlflow tracking server."""
    df = pd.read_parquet(run.info.artifact_uri + artifact_name)
    return df


def _log_artifact_as_parquet(
    artifact: pd.DataFrame, artifact_name: str, index: bool = True
):
    """Upload a DataFrame as a CSV to mlflow tracking server."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_path = Path(tmp_dir) / artifact_name
        artifact.to_parquet(parquet_path, index=index)
        return mlflow.log_artifact(parquet_path, artifact_name)


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
        extracted = _load_artifact_as_parquet(most_recent_run, "/extracted.parquet")
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


def extract_filings(
    dataset: str,
    filings_to_extract: pd.DataFrame,
    extraction_metadata: pd.DataFrame,
    extracted: pd.DataFrame,
    num_filings: int,
    experiment_name: str,
    cloud_interface: GCSArchive,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Extract filings in `filings_to_extract`."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id):
        # Extract data for desired filings
        if dataset == "basic_10k":
            extraction_metadata, extracted = basic_10k.extract(
                filings_to_extract,
                extraction_metadata,
                extracted,
                cloud_interface,
            )
        else:
            model_checkpoint = load_model()
            model = model_checkpoint["model"]
            processor = model_checkpoint["tokenizer"]
            # populate extraction metadata with filenames
            # TODO: does extraction md already have filenames in it? check this
            extraction_metadata = pd.concat(
                [
                    extraction_metadata,
                    pd.DataFrame(
                        {
                            "filename": filings_to_extract["filename"].unique(),
                            "success": False,
                        }
                    ).set_index("filename"),
                ]
            )
            # TODO: there's probably a faster way to do this with less caching
            with tempfile.TemporaryDirectory() as temp_dir:
                # get Sec10K objects
                # TODO: does it save time if we don't cache them?
                temp_dir = Path(temp_dir)
                cloud_interface.get_filings(
                    filings_to_extract, cache_directory=temp_dir, cache_pdf=True
                )
                _, _, extracted, extraction_metadata = perform_inference(
                    pdfs_dir=temp_dir,
                    model=model,
                    processor=processor,
                    extraction_metadata=extraction_metadata,
                )
            extracted["filename"] = extracted["id"].apply(get_metadata_filename)
            extracted = extracted.set_index("filename")

        # Use metadata to log generic metrics
        extraction_metadata = ExtractionMetadataSchema.validate(extraction_metadata)
        mlflow.log_metrics(
            {
                "num_failed": (~extraction_metadata["success"]).sum(),
                "ratio_extracted": len(extraction_metadata) / num_filings,
            }
        )

        # Log the extraction results + metadata for future reference/analysis
        _log_artifact_as_csv(extraction_metadata, "extraction_metadata.csv")
        _log_artifact_as_parquet(extracted, "extracted.parquet")
    logger.info(
        f"Finished extracting {len(extraction_metadata)} filings from {dataset}."
    )
    return extracted


class ExtractConfig(ConfigurableResource):
    """Basic configuration for an extraction run."""

    num_filings: int = -1


def extract_asset_factory(dataset: str) -> asset:
    """Produce asset to extract `dataset`."""

    @asset(
        name=f"{dataset}_extract",
        required_resource_keys={
            f"{dataset}_extract_config",
            f"{dataset}_extract_mlflow",
            "cloud_interface",
        },
    )
    def extract(context) -> pd.DataFrame:
        config = context.resources.original_resource_dict[f"{dataset}_extract_config"]
        cloud_interface: GCSArchive = context.resources.cloud_interface
        mlflow_interface = context.resources.original_resource_dict[
            f"{dataset}_extract_mlflow"
        ]
        experiment_name = mlflow_interface.experiment_name
        metadata = cloud_interface.get_metadata()

        # Get filings to extract as well as any existing metadata for run
        filings_to_extract, extraction_metadata, extracted, run_id = (
            _get_filings_to_extract(
                experiment_name,
                metadata,
                continue_run=mlflow_interface.continue_run,
                num_filings=config.num_filings,
            )
        )

        return extract_filings(
            dataset=dataset,
            filings_to_extract=filings_to_extract,
            extraction_metadata=extraction_metadata,
            extracted=extracted,
            num_filings=len(metadata),
            experiment_name=experiment_name,
            cloud_interface=cloud_interface,
            run_id=run_id,
        )

    return extract


def jaccard_similarity(
    computed_df: pd.DataFrame, validation_df: pd.DataFrame, value_col: str
) -> float:
    """Get the Jaccard similarity between two Series.

    Calculated as the intersection of the set divided
    by the union of the set.

    Args:
        computed_df: Extracted data.
        validation_df: Expected extraction results.
        value_col: Column to calculate Jaccard similarity on.
            Must be present in both dataframes.
    """
    # fill nans to make similarity comparison more accurate
    if (computed_df[value_col].dtype == float) and (
        validation_df[value_col].dtype == float
    ):
        computed_df[value_col] = computed_df[value_col].fillna(999)
        validation_df[value_col] = validation_df[value_col].fillna(999)
    else:
        computed_df[value_col] = computed_df[value_col].fillna("zzz")
        validation_df[value_col] = validation_df[value_col].fillna("zzz")
    intersection = set(computed_df[value_col]).intersection(
        set(validation_df[value_col])
    )
    union = set(computed_df[value_col]).union(set(validation_df[value_col]))
    return float(len(intersection)) / float(len(union))


def compute_ex21_validation_metrics(
    computed_df: pd.DataFrame, validation_df: pd.DataFrame
):
    """Compute validation metrics for Ex. 21 extraction."""
    shared_cols = validation_df.columns.intersection(computed_df.columns)
    validation_df = validation_df.astype(computed_df[shared_cols].dtypes)
    n_equal = 0
    validation_filenames = validation_df["id"].unique()
    n_files = len(validation_filenames)
    table_metrics_dict = {}
    jaccard_dict = {}
    incorrect_files = []
    # iterate through each file and check each extracted table
    for filename in validation_filenames:
        extracted_table_df = computed_df[computed_df["id"] == filename].reset_index(
            drop=True
        )
        validation_table_df = validation_df[
            validation_df["id"] == filename
        ].reset_index(drop=True)
        # check if the tables are exactly equal
        if extracted_table_df.equals(validation_table_df):
            # TODO: strip llc and other company strings before comparison
            n_equal += 1
        else:
            incorrect_files.append(filename)
        # compute precision and recall for each column
        table_metrics_dict[filename] = {}
        jaccard_dict[filename] = {}
        for col in ["subsidiary", "loc", "own_per"]:
            table_prec_recall = compute_validation_metrics(
                extracted_table_df, validation_table_df, value_col=col
            )
            table_metrics_dict[filename][f"{col}_precision"] = table_prec_recall[
                "precision"
            ]
            table_metrics_dict[filename][f"{col}_recall"] = table_prec_recall["recall"]
            # get the jaccard similarity between columns
            jaccard_dict[filename][col] = jaccard_similarity(
                computed_df=extracted_table_df,
                validation_df=validation_table_df,
                value_col=col,
            )

    jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient="index").reset_index()
    prec_recall_df = pd.DataFrame.from_dict(
        table_metrics_dict, orient="index"
    ).reset_index()
    _log_artifact_as_csv(
        jaccard_df,
        artifact_name="jaccard_per_table.csv",
    )
    _log_artifact_as_csv(
        prec_recall_df,
        artifact_name="precision_recall_per_table.csv",
    )
    _log_artifact_as_csv(
        pd.DataFrame({"filename": incorrect_files}),
        artifact_name="incorrect_filenames.csv",
    )
    return {
        "table_accuracy": n_equal / n_files,
        "avg_subsidiary_jaccard_sim": jaccard_df["subsidiary"].sum() / n_files,
        "avg_location_jaccard_sim": jaccard_df["loc"].sum() / n_files,
        "avg_own_per_jaccard_sim": jaccard_df["own_per"].sum() / n_files,
        "avg_subsidiary_precision": prec_recall_df["subsidiary_precision"].sum()
        / n_files,
        "avg_location_precision": prec_recall_df["loc_precision"].sum() / n_files,
        "avg_own_per_precision": prec_recall_df["own_per_precision"].sum() / n_files,
        "avg_subsidiary_recall": prec_recall_df["subsidiary_recall"].sum() / n_files,
        "avg_location_recall": prec_recall_df["loc_recall"].sum() / n_files,
        "avg_own_per_recall": prec_recall_df["own_per_recall"].sum() / n_files,
    }


def clean_ex21_validation_set(validation_df: pd.DataFrame):
    """Clean Ex. 21 validation data to match extracted format."""
    validation_df = validation_df.rename(
        columns={
            "Filename": "id",
            "Subsidiary": "subsidiary",
            "Location of Incorporation": "loc",
            "Ownership Percentage": "own_per",
        }
    )
    validation_df["own_per"] = validation_df["own_per"].astype(str)
    validation_df["filename"] = validation_df["id"].apply(get_metadata_filename)
    validation_df = clean_extracted_df(validation_df)
    return validation_df


def validate_extraction(
    dataset: str, experiment_name: str, cloud_interface: GCSArchive
):
    """Run extraction on validation set and compare results to labeled data."""
    validation_set = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data") / f"{dataset}_labels.csv"
    )
    if dataset == "ex21":
        validation_set = clean_ex21_validation_set(validation_set)

    # Get metadata for labelled filings
    to_extract = cloud_interface.get_metadata(
        filenames=list(validation_set["filename"])
    )

    # Get filings to extract as well as any existing metadata for run
    filings_to_extract, extraction_metadata, extracted, run_id = (
        _get_filings_to_extract(
            experiment_name,
            to_extract,
        )
    )

    # Extract data from filings
    extracted = extract_filings(
        dataset=dataset,
        filings_to_extract=filings_to_extract,
        extraction_metadata=extraction_metadata,
        extracted=extracted,
        num_filings=len(to_extract),
        experiment_name=experiment_name,
        cloud_interface=cloud_interface,
        run_id=run_id,
    )

    # Set index for validation set based on returned extracted DF
    validation_set = validation_set.set_index(extracted.index.names)

    # Get extraction run from mlflow and start again to log validation metrics
    run = _get_most_recent_run(experiment_name)
    with mlflow.start_run(run_id=run.info.run_id):
        # Compute metrics and log
        if dataset == "basic_10k":
            mlflow.log_metrics(
                compute_validation_metrics(extracted, validation_set, "value")
            )
        else:
            mlflow.log_metrics(
                compute_ex21_validation_metrics(extracted, validation_set)
            )
        # Log validation set used to compute metrics
        _log_artifact_as_csv(validation_set, "labels.csv")


def validate_extraction_asset_factory(dataset: str):
    """Create asset that extracts validation filings and compute validation metrics."""

    @asset(
        name=f"{dataset}_extract_validate",
        required_resource_keys={
            f"{dataset}_extract_validate_mlflow",
            "cloud_interface",
        },
    )
    def validate(context):
        cloud_interface: GCSArchive = context.resources.cloud_interface
        experiment_name = context.resources.original_resource_dict[
            f"{dataset}_extract_validate_mlflow"
        ].experiment_name
        return validate_extraction(dataset, experiment_name, cloud_interface)

    return validate


basic_10k_extract = extract_asset_factory("basic_10k")
basic_10k_validate = validate_extraction_asset_factory("basic_10k")
ex21_extract = extract_asset_factory("ex21")
ex21_validate = validate_extraction_asset_factory("ex21")
