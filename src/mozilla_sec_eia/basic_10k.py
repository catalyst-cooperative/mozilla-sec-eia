"""Implement functions for handling data from basic 10k filings (not exhibit 21)."""

import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from mozilla_sec_eia.utils import GCSArchive

logger = logging.getLogger(f"catalystcoop.{__name__}")


def extract():
    """Extract basic 10K data and write to postgres table."""

    def extract_10k(filing):
        return filing.extract_company_data()

    logger.info("Starting basic 10K extraction.")
    archive = GCSArchive()
    metadata = archive.get_metadata()

    extracted = []
    with ProcessPoolExecutor() as executor:
        for i, ext in enumerate(
            executor.map(extract_10k, archive.iterate_filings(metadata))
        ):
            extracted += ext
            if (i % 10) == 0:
                logger.info(f"Extracted {i}/{len(metadata)}")

    df = pd.DataFrame(extracted)
    df.to_sql(
        "basic_10k",
        archive._engine,
        method="multi",
        if_exists="replace",
        index=False,
    )
