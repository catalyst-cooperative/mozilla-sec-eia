"""Implement top level extraction methods and tooling."""

import io
import json
import logging
import re
import tempfile
from importlib import resources
from pathlib import Path

import mlflow
import pandas as pd
import pandera as pa
from mlflow.entities import Run

from mozilla_sec_eia import basic_10k
from mozilla_sec_eia.ex_21.inference import clean_extracted_df, perform_inference
from mozilla_sec_eia.utils.cloud import (
    GCSArchive,
    get_metadata_filename,
    initialize_mlflow,
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


def compute_precision_and_recall(
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
    intersection = set(computed_df[value_col]).intersection(
        set(validation_df[value_col])
    )
    union = set(computed_df[value_col]).union(set(validation_df[value_col]))
    return float(len(intersection)) / float(len(union))


def strip_down_company_names(ser: pd.Series) -> pd.Series:
    """Strip LLC and other company name suffixes from a column."""
    # this JSON is taken from PUDL package data (used for CompanyNameCleaner)
    json_source = (
        resources.files("mozilla_sec_eia.package_data") / "us_legal_forms.json"
    )
    with json_source.open() as json_file:
        legal_terms_dict = json.load(json_file)["legal_forms"]["en"]
    terms_list = list(legal_terms_dict.keys())
    for key in legal_terms_dict:
        terms_list += legal_terms_dict[key]
    reg = r"\b(?:" + "|".join(re.escape(word) for word in terms_list) + r")\b"
    ser = ser.replace(reg, "", regex=True)
    ser = ser.str.strip()
    # strip commas or other special chars that might be at the end of the name
    ser = ser.str.replace(r"[^[^\w&\s]+|[^\w&\s]+$]+", "", regex=True)
    ser = ser.str.strip()
    return ser


def _fill_nulls_for_comparison(ser: pd.Series):
    """Fill nans in column to make similarity comparison more accurate."""
    ser = ser.fillna(999) if ser.dtype == float else ser.fillna("zzz")
    return ser


def compute_ex21_validation_metrics(
    computed_df: pd.DataFrame, validation_df: pd.DataFrame
):
    """Compute validation metrics for Ex. 21 extraction."""
    shared_cols = validation_df.columns.intersection(computed_df.columns)
    validation_df = validation_df.astype(computed_df[shared_cols].dtypes)
    # strip llc and other company name parts for the similarity comparison
    computed_df["subsidiary"] = strip_down_company_names(computed_df["subsidiary"])
    validation_df["subsidiary"] = strip_down_company_names(validation_df["subsidiary"])
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
            n_equal += 1
        else:
            incorrect_files.append(filename)
        # compute jaccard sim + precision and recall for each column
        table_metrics_dict[filename] = {}
        jaccard_dict[filename] = {}
        for col in ["subsidiary", "loc", "own_per"]:
            extracted_table_df[col] = _fill_nulls_for_comparison(
                extracted_table_df[col]
            )
            validation_table_df[col] = _fill_nulls_for_comparison(
                validation_table_df[col]
            )
            table_prec_recall = compute_precision_and_recall(
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


# TODO: move this validation into a separate validate module
# and keep this module for extraction?
def validate_extraction(dataset: str):
    """Run extraction on validation set and compare results to labeled data."""
    if dataset not in DATASETS:
        raise RuntimeError(
            f"{dataset} is not a valid dataset. Must be 'ex21' or 'basic_10k'."
        )

    validation_set = pd.read_csv(
        resources.files("mozilla_sec_eia.package_data") / f"{dataset}_labels.csv"
    )

    # Get metadata for labelled filings
    archive = GCSArchive()
    if dataset == "ex21":
        validation_set = clean_ex21_validation_set(validation_set)
    to_extract = archive.get_metadata(filenames=list(validation_set["filename"]))
    n_docs = len(validation_set["filename"].unique())

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
            mlflow.log_param("validation_set_size", n_docs)
            mlflow.log_metrics(
                compute_precision_and_recall(extracted, validation_set, "value")
            )
        else:
            mlflow.log_param("validation_set_size", n_docs)
            mlflow.log_metrics(
                compute_ex21_validation_metrics(extracted, validation_set)
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
    if dataset not in DATASETS:
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
    filings_to_extract_md, extraction_metadata, extracted, run_id = (
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
                filings_to_extract_md,
                extraction_metadata,
                extracted,
                archive,
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
                            "filename": filings_to_extract_md["filename"].unique(),
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
                archive.get_filings(
                    filings_to_extract_md, cache_directory=temp_dir, cache_pdf=True
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
