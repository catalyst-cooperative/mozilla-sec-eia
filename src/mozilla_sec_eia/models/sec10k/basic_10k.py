"""Implement functions for handling data from basic 10k filings (not exhibit 21)."""

import logging

import pandas as pd
from dagster import AssetIn, asset

from mozilla_sec_eia.library import validation_helpers

from .extract import (
    Sec10kExtractor,
    sec10k_extraction_asset_factory,
    sec10k_filing_metadata,
)
from .utils.cloud import GCSArchive, Sec10K, cloud_interface_resource

logger = logging.getLogger(f"catalystcoop.{__name__}")


class Basic10kExtractor(Sec10kExtractor):
    """Implement Sec10kExtractor for basic 10k company info data."""

    name: str = "basic_10k_extractor"

    def _extract_10k(self, filing: Sec10K):
        """Extract basic company data from filing."""
        logger.info(f"Extracting 10K company data from filing: {filing.filename}")
        header = True
        current_block = None
        values = []
        filer_count = 0
        block_counts = {
            "company data": 0,
            "filing values": 0,
            "business address": 0,
            "mail address": 0,
            "former company": 0,
        }
        unmatched_keys = []
        for line in filing.filing_text.splitlines():
            match line.replace("\t", "").lower().split(":"):
                case ["filer", ""]:
                    filer_count += 1
                    header = False
                case [
                    (
                        "company data"
                        | "filing values"
                        | "business address"
                        | "mail address"
                        | "former company"
                    ) as block,
                    "",
                ] if not header:
                    current_block = block
                    block_counts[current_block] += 1
                case [key, ""] if current_block is not None:
                    key = f"{block}_{key}".replace(" ", "_")
                    logger.warning(
                        f"No value found for {key} for filing {filing.filename}"
                    )
                    unmatched_keys.append(key)
                case [key, value] if current_block is not None:
                    key = key.replace(" ", "_")
                    values.append(
                        {
                            "filename": filing.filename,
                            "filer_count": filer_count - 1,
                            "block": current_block.replace(" ", "_"),
                            "block_count": block_counts[current_block] - 1,
                            "key": key.replace(" ", "_"),
                            "value": value,
                        }
                    )
                case ["</sec-header>" | "</ims-header>"]:
                    break
                case _ if header:
                    continue

        return pd.DataFrame(values), filing.filename, unmatched_keys

    def extract_filings(
        self,
        filings_to_extract: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract basic 10K data and return extracted data/metadata."""
        logger.info("Starting basic 10K extraction.")
        logger.info(f"Extracting {len(filings_to_extract)} filings.")

        extraction_metadata = pd.DataFrame(
            {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
        ).set_index("filename")
        extracted = pd.DataFrame()

        for filing in self.cloud_interface.iterate_filings(filings_to_extract):
            ext, filename, unmatched_keys = self._extract_10k(filing)
            extraction_metadata.loc[filename, ["success", "unmatched_keys"]] = [
                len(ext) > 0,
                ",".join(unmatched_keys),
            ]
            extracted = pd.concat([extracted, ext])

        return (
            extraction_metadata,
            extracted.set_index(
                ["filename", "filer_count", "block", "block_count", "key"]
            ),
        )


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


basic_10k_extractor_resource = Basic10kExtractor(
    cloud_interface=cloud_interface_resource
)
basic_10k_production_extraction = sec10k_extraction_asset_factory(
    "basic_10k",
    basic_10k_extractor_resource,
    extraction_metadata_asset_name="basic_10k_extraction_metadata",
    extracted_asset_name="basic_10k_company_info",
)


basic_10k_validation_extraction = sec10k_extraction_asset_factory(
    "basic_10k_validation",
    basic_10k_extractor_resource,
    filing_metadata_asset_name="basic_10k_validation_filing_metadata",
    extraction_metadata_asset_name="basic_10k_extraction_validation_metadata",
    extracted_asset_name=basic_10k_extracted_validation_asset_name,
    partitions_def=None,
    io_manager_key="mlflow_pandas_artifact_io_manager",
)

production_assets = [basic_10k_production_extraction, sec10k_filing_metadata]

validation_assets = [
    basic_10k_validation_extraction,
    basic_10k_validation_set,
    basic_10k_validation_filing_metadata,
    basic_10k_extraction_validation_metrics,
]
