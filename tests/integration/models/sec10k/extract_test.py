"""Validate basic 10k and exhibit 21 extraction."""

import logging
import unittest

import dotenv
import numpy as np
import pandas as pd
import pytest
from dagster import EnvVar, build_asset_context
from mozilla_sec_eia.ex_21.inference import (
    clean_extracted_df,
    create_inference_dataset,
    perform_inference,
)
from mozilla_sec_eia.extract import (
    _get_most_recent_run,
    basic_10k_validate,
    ex21_validate,
)
from mozilla_sec_eia.utils.cloud import GCSArchive, MlflowInterface
from mozilla_sec_eia.utils.layoutlm import load_model
from pandas.testing import assert_frame_equal

logger = logging.getLogger(f"catalystcoop.{__name__}")


def test_basic_10k_extraction():
    """Run full 10k extraction on validation set and verify desired metrics are met."""
    dotenv.load_dotenv()
    experiment_name = "basic_10k_validate_test"
    cloud_interface = GCSArchive(
        filings_bucket_name=EnvVar("GCS_FILINGS_BUCKET_NAME"),
        labels_bucket_name=EnvVar("GCS_LABELS_BUCKET_NAME"),
        metadata_db_instance_connection=EnvVar("GCS_METADATA_DB_INSTANCE_CONNECTION"),
        user=EnvVar("GCS_IAM_USER"),
        metadata_db_name=EnvVar("GCS_METADATA_DB_NAME"),
        project=EnvVar("GCS_PROJECT"),
    )

    with build_asset_context(
        resources={
            "basic_10k_extract_validate_mlflow": MlflowInterface(
                experiment_name=experiment_name,
                continue_run=False,
                tracking_uri="sqlite:///:memory:",
                cloud_interface=cloud_interface,
            ),
            "cloud_interface": cloud_interface,
        }
    ) as context:
        basic_10k_validate(context)
    run = _get_most_recent_run(experiment_name)

    assert run.data.metrics["precision"] == 1
    assert run.data.metrics["recall"] == 1


@pytest.mark.xfail
def test_ex21_validation(test_mlflow_init_func):
    """Run full Ex. 21 extraction on validation set and verify metrics are met."""
    dotenv.load_dotenv()
    experiment_name = "ex21_validate_test"
    cloud_interface = GCSArchive(
        filings_bucket_name=EnvVar("GCS_FILINGS_BUCKET_NAME"),
        labels_bucket_name=EnvVar("GCS_LABELS_BUCKET_NAME"),
        metadata_db_instance_connection=EnvVar("GCS_METADATA_DB_INSTANCE_CONNECTION"),
        user=EnvVar("GCS_IAM_USER"),
        metadata_db_name=EnvVar("GCS_METADATA_DB_NAME"),
        project=EnvVar("GCS_PROJECT"),
    )

    with build_asset_context(
        resources={
            "ex21_validate_test": MlflowInterface(
                experiment_name=experiment_name,
                continue_run=False,
                tracking_uri="sqlite:///:memory:",
                cloud_interface=cloud_interface,
            ),
            "cloud_interface": cloud_interface,
        }
    ) as context:
        ex21_validate(context)
    run = _get_most_recent_run(experiment_name)
    # TODO: add in actual metric checks once validation is ready
    assert run.data.metrics["ratio_extracted"] == 1


@pytest.fixture
def model_checkpoint():
    """Load model from tracking server and return."""
    return load_model()


def test_model_loading(model_checkpoint):
    """Test loading a fine-tuned LayoutLM model from MLFlow."""
    assert "model" in model_checkpoint
    assert "tokenizer" in model_checkpoint


def test_dataset_creation(test_dir):
    pdf_dir = test_dir / "data/test_pdfs"
    dataset = create_inference_dataset(pdfs_dir=pdf_dir)
    assert dataset.shape == (2, 4)


def test_ex21_inference_and_table_extraction(
    test_dir, test_mlflow_init_func, model_checkpoint
):
    """Test performing inference and extracting an Ex. 21 table."""
    model = model_checkpoint["model"]
    processor = model_checkpoint["tokenizer"]
    pdf_dir = test_dir / "data" / "test_pdfs"
    extraction_metadata = pd.DataFrame(
        {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
    ).set_index("filename")
    with unittest.mock.patch("mozilla_sec_eia.extract.initialize_mlflow"):
        test_mlflow_init_func()
        logit_list, pred_list, output_df, extraction_metadata = perform_inference(
            pdfs_dir=pdf_dir,
            model=model,
            processor=processor,
            extraction_metadata=extraction_metadata,
            device="cpu",
        )
    # we don't normally want to sort by id and subsidiary
    # but sort here for the sake of just testing whether dataframe
    # row values are the same without worrying about order
    output_df = output_df.sort_values(by=["id", "subsidiary"]).reset_index(drop=True)
    # TODO: uncomment with new model checkpoint and 7th label included
    # assert logit_list[0].shape == (1, 512, len(LABELS))
    expected_out_path = test_dir / "data" / "inference_and_extraction_expected_out.csv"
    expected_out_df = pd.read_csv(
        expected_out_path,
        dtype={"id": str, "subsidiary": str, "loc": str, "own_per": np.float64},
    )
    expected_out_df["own_per"] = expected_out_df["own_per"].astype(str)
    expected_out_df = clean_extracted_df(expected_out_df)
    expected_out_df = expected_out_df.sort_values(by=["id", "subsidiary"]).reset_index(
        drop=True
    )
    assert_frame_equal(expected_out_df, output_df, check_like=True)