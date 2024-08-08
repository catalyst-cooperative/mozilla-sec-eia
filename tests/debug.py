from mozilla_sec_eia.utils.layoutlm import load_model
from mozilla_sec_eia.ex_21.inference import perform_inference
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path

model_checkpoint = load_model()
model = model_checkpoint["model"]
processor = model_checkpoint["tokenizer"]
test_dir = Path(__file__).parent
pdf_dir = test_dir / "data" / "test_pdfs"
logit_list, pred_list, output_df = perform_inference(
    pdfs_dir=pdf_dir,
    model=model,
    processor=processor,
    device="cpu",
)
output_df = output_df.sort_values(by="id")
expected_out_path = test_dir / "data" / "inference_and_extraction_expected_out.csv"
expected_out_df = pd.read_csv(expected_out_path)
expected_out_df = expected_out_df.sort_values(by="id")
assert_frame_equal(expected_out_df, output_df, check_like=True)
