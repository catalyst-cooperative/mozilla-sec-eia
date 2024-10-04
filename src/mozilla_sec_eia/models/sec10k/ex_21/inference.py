"""Module for formatting inputs and performing inference with a fine-tuned LayoutLM model."""

import logging
import traceback

import pandas as pd
from mlflow.pyfunc import PyFuncModel

from ..entities import Ex21CompanyOwnership
from ..utils.cloud import GCSArchive
from .data.inference import create_inference_dataset

logger = logging.getLogger(f"catalystcoop.{__name__}")


def extract_filings(
    filings: pd.DataFrame,
    layoutlm: PyFuncModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create huggingface dataset from filings and perform extraction."""
    try:
        failed_metadata, dataset = create_inference_dataset(
            filing_metadata=filings,
            cloud_interface=GCSArchive(),
            has_labels=False,
        )
        metadata, extracted = layoutlm.predict(dataset)
        metadata = pd.concat([failed_metadata, metadata])
    except Exception as e:
        logger.warning(traceback.format_exc())
        logger.warning(f"Error while extracting filings: {filings.index}")
        metadata = pd.DataFrame(
            {
                "filename": filings.index,
                "success": [False] * len(filings),
                "notes": [str(e)] * len(filings),
            }
        ).set_index("filename")
        extracted = Ex21CompanyOwnership.example(size=0)
    return metadata, extracted
