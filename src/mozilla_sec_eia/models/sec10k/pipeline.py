"""Implement top level extraction methods and tooling."""

import logging

import pandas as pd
import pandera as pa
from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetOut,
    StaticPartitionsDefinition,
    asset,
    multi_asset,
    with_resources,
)

from mozilla_sec_eia.library import validation_helpers
from mozilla_sec_eia.library.mlflow import (
    mlflow_production_interface,
    mlflow_train_test_interface,
)
from mozilla_sec_eia.library.pipeline import (
    create_production_pipeline,
    create_validation_pipeline,
)

from .basic_10k import Basic10kExtractor
from .ex_21.inference import Exhibit21Extractor, clean_extracted_df
from .extract import Sec10kExtractor
from .utils.cloud import GCSArchive, cloud_interface_resource, get_metadata_filename
from .utils.layoutlm import LayoutlmResource

logger = logging.getLogger(f"catalystcoop.{__name__}")

DATASETS = ["ex21", "basic_10k"]


class ExtractionMetadataSchema(pa.DataFrameModel):
    """Define the required schema for extraction metadata.

    Extra columns are permitted, but these are required for computing extraction metrics.
    """

    filename: pa.typing.Index[str] = pa.Field(check_name=True)
    success: bool = pa.Field(coerce=True)


# Create year_quarter partitions
partitions_def = StaticPartitionsDefinition(
    [f"{year}q{quarter}" for year in range(1994, 2024) for quarter in range(1, 5)]
)


@asset(partitions_def=partitions_def)
def sec10k_filing_metadata(
    context: AssetExecutionContext,
    cloud_interface: GCSArchive,
) -> pd.DataFrame:
    """Return filing metadata for year_quarter partition."""
    year_quarter = context.partition_key
    df = cloud_interface.get_metadata(year_quarter=year_quarter)
    return df


def sec10k_extraction_asset_factory(
    name: str,
    sec10k_extractor: Sec10kExtractor,
    partitions_def=None,
    filing_metadata_asset_name: str = "sec10k_filing_metadata",
    extraction_metadata_asset_name: str = "extraction_metadata",
    extracted_asset_name: str = "extraction_metadata",
):
    """Create asset to extract data from sec10k data.

    Args:
        name: Name of extraction asset.
        sec10k_extractor: Subclass of Sec10kExtractor used to extract data.
        partitions_def: Partitions for asset (production uses year_quarter parts,
            validation is not partitioned.
        filing_metadata_asset_name: Name of input asset with metadata of filings to
            extract.
        extraction_metadata_asset_name: Name of output asset containing metadata
            from extraction run.
        extracted_asset_name: Name of output asset containing extracted data.
    """

    @multi_asset(
        name=name,
        outs={
            extraction_metadata_asset_name: AssetOut(),
            extracted_asset_name: AssetOut(),
        },
        ins={"sec10k_filing_metadata": AssetIn(filing_metadata_asset_name)},
        partitions_def=partitions_def,
        required_resource_keys={sec10k_extractor.name},
    )
    def extract_filings(
        context: AssetExecutionContext, sec10k_filing_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run Sec10kExtractor on selected partition and return."""
        extractor = context.resources.original_resource_dict[sec10k_extractor.name]
        extraction_metadata, extracted = extractor.extract_filings(
            sec10k_filing_metadata
        )
        return extraction_metadata, extracted

    return with_resources([extract_filings], {sec10k_extractor.name: sec10k_extractor})[
        0
    ]


@asset
def basic_10k_validation_set() -> pd.DataFrame:
    """Return dataframe containing basic 10k validation data."""
    return validation_helpers.load_validation_data(
        "basic_10k_labels.csv",
        index_cols=["filename", "filer_count", "block", "block_count", "key"],
    )


basic_10k_extracted_validation_asset_name = "basic_10k_company_info_validation"


@asset(
    ins={
        basic_10k_extracted_validation_asset_name: AssetIn(
            basic_10k_extracted_validation_asset_name
        ),
        "basic_10k_validation_set": AssetIn(),
    },
    io_manager_key="mlflow_metrics_io_manager",
)
def basic_10k_extraction_validation_metrics(**kwargs):
    """Compute basic 10k extraction validation metrics."""
    computed = kwargs[basic_10k_extracted_validation_asset_name]
    validation = kwargs["basic_10k_validation_set"]

    return validation_helpers.pandas_compute_precision_recall(
        computed, validation, value_col="value"
    )


@asset
def basic_10k_validation_filing_metadata(
    cloud_interface: GCSArchive,
    basic_10k_validation_set: pd.DataFrame,
) -> pd.DataFrame:
    """Get sec 10k filing metadata from validation set."""
    filing_metadata = cloud_interface.get_metadata()
    return filing_metadata[
        filing_metadata["filename"].isin(
            basic_10k_validation_set.index.get_level_values("filename").unique()
        )
    ]


# Register basic 10k extraction pipeline
create_production_pipeline(
    "basic_10k_extraction",
    [
        sec10k_filing_metadata,
        sec10k_extraction_asset_factory(
            "basic_10k",
            Basic10kExtractor(cloud_interface=cloud_interface_resource),
            partitions_def=partitions_def,
            extraction_metadata_asset_name="basic_10k_extraction_metadata",
            extracted_asset_name="basic_10k_company_info",
        ),
    ],
    resources={"cloud_interface": cloud_interface_resource},
)


# Register basic 10k extraction validation pipeline
create_validation_pipeline(
    "basic_10k_extraction",
    [
        basic_10k_validation_filing_metadata,
        sec10k_extraction_asset_factory(
            "basic_10k",
            Basic10kExtractor(cloud_interface=cloud_interface_resource),
            filing_metadata_asset_name="basic_10k_validation_filing_metadata",
            extraction_metadata_asset_name="basic_10k_extraction_validation_metadata",
            extracted_asset_name=basic_10k_extracted_validation_asset_name,
        ),
        basic_10k_validation_set,
        basic_10k_extraction_validation_metrics,
    ],
    resources={"cloud_interface": cloud_interface_resource},
)


@asset
def ex21_validation_set() -> pd.DataFrame:
    """Return dataframe containing basic 10k validation data."""
    return clean_ex21_validation_set(
        validation_helpers.load_validation_data("ex21_labels.csv")
    )


@asset
def ex21_validation_filing_metadata(
    cloud_interface: GCSArchive,
    ex21_validation_set: pd.DataFrame,
) -> pd.DataFrame:
    """Get sec 10k filing metadata from validation set."""
    filing_metadata = cloud_interface.get_metadata()
    return filing_metadata[
        filing_metadata["filename"].isin(
            ex21_validation_set.index.get_level_values("filename").unique()
        )
    ]


ex21_extracted_validation_asset_name = "ex21_validation"


@multi_asset(
    ins={
        "computed_df": AssetIn(ex21_extracted_validation_asset_name),
        "validation_df": AssetIn("ex21_validation_set"),
    },
    outs={
        "ex21_jaccard_per_table": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager"
        ),
        "ex21_precision_recall_per_table": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager"
        ),
        "ex21_incorrect_filenames": AssetOut(
            io_manager_key="mlflow_pandas_artifact_io_manager"
        ),
        "ex21_extraction_metrics": AssetOut(io_manager_key="mlflow_metrics_io_manager"),
    },
)
def ex21_validation_metrics(computed_df: pd.DataFrame, validation_df: pd.DataFrame):
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
            table_prec_recall = validation_helpers.pandas_compute_precision_recall(
                extracted_table_df, validation_table_df, value_col=col
            )
            table_metrics_dict[filename][f"{col}_precision"] = table_prec_recall[
                "precision"
            ]
            table_metrics_dict[filename][f"{col}_recall"] = table_prec_recall["recall"]
            # get the jaccard similarity between columns
            jaccard_dict[filename][col] = validation_helpers.jaccard_similarity(
                computed_df=extracted_table_df,
                validation_df=validation_table_df,
                value_col=col,
            )

    jaccard_df = pd.DataFrame.from_dict(jaccard_dict, orient="index").reset_index()
    prec_recall_df = pd.DataFrame.from_dict(
        table_metrics_dict, orient="index"
    ).reset_index()

    return (
        jaccard_df,
        prec_recall_df,
        pd.DataFrame({"filename": incorrect_files}),
        {
            "table_accuracy": n_equal / n_files,
            "avg_subsidiary_jaccard_sim": jaccard_df["subsidiary"].sum() / n_files,
            "avg_location_jaccard_sim": jaccard_df["loc"].sum() / n_files,
            "avg_own_per_jaccard_sim": jaccard_df["own_per"].sum() / n_files,
            "avg_subsidiary_precision": prec_recall_df["subsidiary_precision"].sum()
            / n_files,
            "avg_location_precision": prec_recall_df["loc_precision"].sum() / n_files,
            "avg_own_per_precision": prec_recall_df["own_per_precision"].sum()
            / n_files,
            "avg_subsidiary_recall": prec_recall_df["subsidiary_recall"].sum()
            / n_files,
            "avg_location_recall": prec_recall_df["loc_recall"].sum() / n_files,
            "avg_own_per_recall": prec_recall_df["own_per_recall"].sum() / n_files,
        },
    )


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


# Register ex21 extraction pipeline
create_production_pipeline(
    "ex21_extraction",
    [
        sec10k_filing_metadata,
        sec10k_extraction_asset_factory(
            "ex21",
            Exhibit21Extractor(
                cloud_interface=cloud_interface_resource,
                layoutlm=LayoutlmResource(mlflow_interface=mlflow_production_interface),
            ),
            partitions_def=partitions_def,
            extraction_metadata_asset_name="ex21_extraction_metadata",
            extracted_asset_name="ex21_company_info",
        ),
    ],
    resources={"cloud_interface": cloud_interface_resource},
)


# Register ex21 extraction validation pipeline
create_validation_pipeline(
    "ex21_extraction",
    [
        ex21_validation_filing_metadata,
        sec10k_extraction_asset_factory(
            "ex21",
            Exhibit21Extractor(
                cloud_interface=cloud_interface_resource,
                layoutlm=LayoutlmResource(mlflow_interface=mlflow_train_test_interface),
            ),
            filing_metadata_asset_name="ex21_validation_filing_metadata",
            extraction_metadata_asset_name="ex21_extraction_validation_metadata",
            extracted_asset_name=ex21_extracted_validation_asset_name,
        ),
        ex21_validation_set,
        ex21_validation_metrics,
    ],
    resources={"cloud_interface": cloud_interface_resource},
)
