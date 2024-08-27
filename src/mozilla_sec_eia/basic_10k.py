"""Implement functions for handling data from basic 10k filings (not exhibit 21)."""

import logging

import pandas as pd
from dagster import Out, op

from mozilla_sec_eia.utils.cloud import GCSArchive, Sec10K

logger = logging.getLogger(f"catalystcoop.{__name__}")
EXPERIMENT_NAME = "basic_10k_extraction"


def _extract_10k(filing: Sec10K):
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
                logger.warning(f"No value found for {key} for filing {filing.filename}")
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


@op(out={"extraction_metadata": Out(), "extracted": Out()})
def extract(
    cloud_interface: GCSArchive,
    filings_to_extract: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract basic 10K data and write to postgres table.

    Args:
        continue_run: If true, only extract filings not in DB, otherwise clobber
            basic_10k table.
    """
    logger.info("Starting basic 10K extraction.")
    logger.info(f"Extracting {len(filings_to_extract)} filings.")

    extraction_metadata = pd.DataFrame(
        {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
    ).set_index("filename")
    extracted = pd.DataFrame()

    for filing in cloud_interface.iterate_filings(filings_to_extract):
        ext, filename, unmatched_keys = _extract_10k(filing)
        extraction_metadata.loc[filename, ["success", "unmatched_keys"]] = [
            len(ext) > 0,
            ",".join(unmatched_keys),
        ]
        extracted = pd.concat([extracted, ext])

    return (
        extraction_metadata,
        extracted.set_index(["filename", "filer_count", "block", "block_count", "key"]),
    )
