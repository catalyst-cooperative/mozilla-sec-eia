"""Integration testing for labeling entities with LayoutLM and extracting Ex. 21 tables."""

import pandas as pd
from mozilla_sec_eia.ex_21.inference import create_inference_dataset, perform_inference
from mozilla_sec_eia.utils.layoutlm import load_model
from pandas.testing import assert_frame_equal


def test_model_loading():
    """Test loading a fine-tuned LayoutLM model from MLFlow."""
    model_checkpoint = load_model()
    assert "model" in model_checkpoint
    assert "tokenizer" in model_checkpoint


def test_dataset_creation(test_dir):
    pdf_dir = test_dir / "data/test_pdfs"
    dataset = create_inference_dataset(pdfs_dir=pdf_dir)
    assert dataset.shape == (2, 4)


def test_inference_and_table_extraction(test_dir):
    """Test performing inference and extracting an Ex. 21 table."""
    model_checkpoint = load_model()
    model = model_checkpoint["model"]
    processor = model_checkpoint["tokenizer"]
    pdf_dir = test_dir / "data" / "test_pdfs"
    logit_list, pred_list, output_df = perform_inference(
        pdfs_dir=pdf_dir,
        model=model,
        processor=processor,
        device="cpu",
    )
    # TODO: uncomment with new model checkpoint and 7th label included
    # assert logit_list[0].shape == (1, 512, len(LABELS))
    expected_out_path = test_dir / "data" / "inference_and_extraction_expected_out.csv"
    expected_out_df = pd.read_csv(expected_out_path)
    assert_frame_equal(expected_out_df, output_df)
