"""Integration testing for labeling entities with LayoutLM and extracting Ex. 21 tables."""

import pandas as pd
from mozilla_sec_eia.ex_21.inference import create_inference_dataset, perform_inference
from mozilla_sec_eia.ex_21.train_extractor import LABELS, load_model
from pandas.testing import assert_frame_equal
from transformers import AutoProcessor


def test_model_loading():
    """Test loading a fine-tuned LayoutLM model from MLFlow."""
    load_model()


def test_dataset_creation(test_dir):
    pdf_dir = test_dir / "data/test_pdfs"
    dataset = create_inference_dataset(pdfs_dir=pdf_dir)
    assert dataset.shape == (2, 4)


def test_inference_and_table_extraction(test_dir):
    """Test performing inference and extracting an Ex. 21 table."""
    model = load_model()
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    pdf_dir = test_dir / "data/test_pdfs"
    logit_list, pred_list, output_df = perform_inference(
        pdfs_dir=pdf_dir,
        model=model,
        processor=processor,
        device="cpu",
    )
    assert logit_list[0].shape == (1, 512, len(LABELS))
    expected_out_path = test_dir / "inference_and_extraction_expected_out.csv"
    expected_out_df = pd.read_csv(expected_out_path)
    assert_frame_equal(expected_out_df, output_df)
