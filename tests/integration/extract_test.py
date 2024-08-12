"""Validate basic 10k and exhibit 21 extraction."""

import logging
import unittest

import numpy as np
import pandas as pd
import pytest
from mozilla_sec_eia.ex_21.inference import create_inference_dataset, perform_inference
from mozilla_sec_eia.extract import (
    _get_experiment_name,
    _get_most_recent_run,
    validate_extraction,
)
from mozilla_sec_eia.utils.layoutlm import load_model
from pandas.testing import assert_frame_equal

logger = logging.getLogger(f"catalystcoop.{__name__}")


def test_basic_10k_extraction(test_mlflow_init_func):
    """Run full 10k extraction on validation set and verify desired metrics are met."""
    with unittest.mock.patch("mozilla_sec_eia.extract.initialize_mlflow"):
        test_mlflow_init_func()
        validate_extraction("basic_10k")
    run = _get_most_recent_run(
        _get_experiment_name("basic_10k", experiment_suffix="validation")
    )

    assert run.data.metrics["precision"] == 1
    assert run.data.metrics["recall"] == 1


def test_ex21_validation(test_mlflow_init_func):
    with unittest.mock.patch("mozilla_sec_eia.extract.initialize_mlflow"):
        test_mlflow_init_func()
        validate_extraction("ex21")
    run = _get_most_recent_run(
        _get_experiment_name("ex21", experiment_suffix="validation")
    )
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


def test_ex21_inference_and_table_extraction(test_dir, model_checkpoint):
    """Test performing inference and extracting an Ex. 21 table."""
    model = model_checkpoint["model"]
    processor = model_checkpoint["tokenizer"]
    pdf_dir = test_dir / "data" / "test_pdfs"
    logit_list, pred_list, output_df = perform_inference(
        pdfs_dir=pdf_dir,
        model=model,
        processor=processor,
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
    expected_out_df = expected_out_df.sort_values(by=["id", "subsidiary"]).reset_index(
        drop=True
    )
    assert_frame_equal(expected_out_df, output_df, check_like=True)
