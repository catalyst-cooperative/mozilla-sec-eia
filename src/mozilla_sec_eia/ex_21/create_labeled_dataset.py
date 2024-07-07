"""Module handling Label Studio inputs and outputs and preparing a dataset for fine-tuning."""

import json
import logging
import os
from pathlib import Path

import pandas as pd

from mozilla_sec_eia.utils.cloud import GCSArchive
from mozilla_sec_eia.utils.pdf import (
    get_pdf_data_from_path,
    normalize_bboxes,
    pil_to_cv2,
    render_page,
)

logger = logging.getLogger(f"catalystcoop.{__name__}")
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


BBOX_COLS = [
    "top_left_x_pdf",
    "top_left_y_pdf",
    "bottom_right_x_pdf",
    "bottom_right_y_pdf",
]


def create_inputs_for_label_studio(
    model_version: str = "v1.0",
    pdfs_dir: Path = ROOT_DIR / "sec10k_filings/pdfs",
    cache_dir: Path = ROOT_DIR / "sec10k_filings",
):
    """Create JSONs and images of Ex. 21 for each doc.

    Text-based PDFs of the documents that are desired for labeling
    should be contained in ``pdf_dir``. The inputs generated by this
    function can be uploaded to the "unlabeled" bucket
    in GCS and then imported into Label Studio. Each JSON represents
    a labeling task and references the corresponding image
    of the Ex. 21.
    """
    # create directories for output task JSONs and images
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir = cache_dir / "jsons"
    json_dir.mkdir(parents=True, exist_ok=True)

    for pdf_filename in os.listdir(pdfs_dir):
        if pdf_filename.split(".")[-1] != "pdf":
            continue
        logger.info(f"Creating JSON for {pdf_filename}")
        src_path = pdfs_dir / pdf_filename
        extracted, pg = get_pdf_data_from_path(src_path)
        txt = extracted["pdf_text"]
        pg_meta = extracted["page"]

        # render an image of the page and save
        full_pg_img = render_page(pg)
        image_filename = pdf_filename.split(".")[0] + ".png"
        full_pg_img.save(image_dir / image_filename)

        # fill in some basic variables
        original_width = pil_to_cv2(full_pg_img).shape[1]
        original_height = pil_to_cv2(full_pg_img).shape[0]
        x_norm = 100 / pg_meta.width_pdf_coord.iloc[0]
        y_norm = 100 / pg_meta.height_pdf_coord.iloc[0]
        # base annotation JSON template
        filename_no_ext = pdf_filename.split(".")[0]
        annotation_json = {
            "id": f"{filename_no_ext}",
            "data": {"ocr": f"gs://labeled-ex21-filings/unlabeled/{image_filename}"},
            "annotations": [],
            "predictions": [{"model_version": f"{model_version}", "result": []}],
        }
        result = []
        # TODO: make more efficient? change to using an apply?
        for i, row in txt.iterrows():
            result += get_bbox_dicts(
                row,
                ind=i,
                x_norm=x_norm,
                y_norm=y_norm,
                original_width=original_width,
                original_height=original_height,
            )

        annotation_json["predictions"][0]["result"] = result
        json_filename = json_dir / Path(filename_no_ext + ".json")
        with Path.open(json_filename, "w") as fp:
            json.dump(annotation_json, fp)


def get_bbox_dicts(
    bbox: pd.Series, ind, x_norm, y_norm, original_width, original_height
) -> list[dict]:
    """Format a dictionary for a bbox in the Label Studio task format."""
    x = bbox["top_left_x_pdf"] * x_norm
    y = bbox["top_left_y_pdf"] * y_norm
    width = (bbox["bottom_right_x_pdf"] - bbox["top_left_x_pdf"]) * x_norm
    height = (bbox["bottom_right_y_pdf"] - bbox["top_left_y_pdf"]) * y_norm
    word = bbox["text"]
    bbox_id = f"bbox_{ind}"
    box_dict = {
        "original_width": original_width,
        "original_height": original_height,
        "image_rotation": 0,
        "value": {"x": x, "y": y, "width": width, "height": height, "rotation": 0},
        "id": bbox_id,
        "from_name": "bbox",
        "to_name": "image",
        "type": "rectangle",
        "origin": "manual",
    }
    word_dict = {
        "original_width": original_width,
        "original_height": original_height,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation": 0,
            "text": [word],
        },
        "id": bbox_id,
        "from_name": "transcription",
        "to_name": "image",
        "type": "textarea",
        "origin": "manual",
    }
    return [box_dict, word_dict]


def _is_cik_in_training_data(labeled_json_filename, tracking_df):
    # TODO: for now CIK is stored as an int, update when fixed
    cik = int(labeled_json_filename.split("/")[-1].split("-")[0])
    return cik in tracking_df.CIK.unique()


# TODO: make this work with GCS input directory not local
# TODO: have default paths?
def format_label_studio_output(
    labeled_json_dir=ROOT_DIR / "sec10k_filings/labeled_jsons",
    pdfs_dir=ROOT_DIR / "sec10k_filings/pdfs",
) -> pd.DataFrame:
    """Format Label Studio output JSONs into dataframe."""
    labeled_df = pd.DataFrame()
    # TODO: make this path stuff less janky?
    tracking_df = pd.read_csv(ROOT_DIR / "labeled_data_tracking.csv")
    for json_filename in os.listdir(labeled_json_dir):
        if not json_filename[0].isdigit():
            continue
        json_file_path = labeled_json_dir / json_filename
        with Path.open(json_file_path) as j:
            doc_dict = json.loads(j.read())
            filename = doc_dict["task"]["data"]["ocr"].split("/")[-1].split(".")[0]
            if not _is_cik_in_training_data(filename, tracking_df=tracking_df):
                continue
            pdf_filename = filename + ".pdf"
            src_path = pdfs_dir / pdf_filename
            extracted, pg = get_pdf_data_from_path(src_path)
            txt = extracted["pdf_text"]
            pg_meta = extracted["page"]
            # normalize bboxes between 0 and 1000 for Hugging Face
            txt = normalize_bboxes(txt_df=txt, pg_meta_df=pg_meta)
            # parse the output dictionary of labeled bounding boxes from Label Studio
            doc_df = pd.DataFrame()
            for item in doc_dict["result"]:
                value = item["value"]
                # sometimes Label Studio will fill in an empty list as a label
                # when there is really no label
                # TODO: do this without dict comprehension?
                if ("labels" in value) and value["labels"] == []:
                    value = {k: v for k, v in value.items() if k != "labels"}
                ind = int(item["id"].split("_")[-1])
                doc_df = pd.concat([doc_df, pd.DataFrame(value, index=[ind])])
            # combine the bounding boxes for each word
            doc_df = doc_df.groupby(level=0).first()
            txt.loc[:, "id"] = filename
            output_df = pd.concat([txt, doc_df[["labels"]]], axis=1)
            labeled_df = pd.concat([labeled_df, output_df])

    # fill in unlabeled words and clean up labeled dataframe
    labeled_df["labels"] = labeled_df["labels"].fillna("O")
    labeled_df = labeled_df.rename(columns={"labels": "ner_tag"})
    non_id_columns = [col for col in labeled_df.columns if col != "id"]
    labeled_df = labeled_df.loc[:, ["id"] + non_id_columns]

    # TODO: add in sanity checks on labeled_df bounding boxes to make sure
    # that no value is above 1000 or below 0

    return labeled_df


def get_image_dict(pdfs_dir):
    """Create a dictionary with filenames and their Ex. 21 images."""
    image_dict = {}
    for pdf_filename in os.listdir(pdfs_dir):
        if pdf_filename.split(".")[-1] != "pdf":
            continue
        pdf_file_path = pdfs_dir / pdf_filename
        _, pg = get_pdf_data_from_path(pdf_file_path)
        full_pg_img = render_page(pg)
        filename = pdf_filename.split(".")[0]
        image_dict[filename] = full_pg_img
    return image_dict


def format_as_ner_annotations(
    labeled_json_path=ROOT_DIR / "sec10k_filings/labeled_jsons",
    pdfs_path=ROOT_DIR / "sec10k_filings/pdfs",
    gcs_folder_name: str = "labeled/",
) -> list[dict]:
    """Format a Label Studio output JSONs as NER annotations.

    Formats the dataframe as named entity recognition annotations.
    # TODO: say more about this format

    Returns:
        ner_annotations: a list of dicts, with one dict for each doc.
    """
    GCSArchive().cache_training_data(labeled_json_path, pdfs_path, gcs_folder_name)

    labeled_df = format_label_studio_output(
        labeled_json_dir=labeled_json_path, pdfs_dir=pdfs_path
    )
    # convert dataframe/dictionary into NER format
    # document_annotation_to_ner https://github.com/butlerlabs/docai/blob/main/docai/annotations/ner_utils.py
    # complete dataset is a list of dicts, with one dict for each doc
    doc_filenames = labeled_df["id"].unique()
    image_dict = get_image_dict(pdfs_dir=pdfs_path)
    ner_annotations = []
    for filename in doc_filenames:
        annotation = {
            "id": filename,
            "tokens": labeled_df.groupby("id")["text"].apply(list).loc[filename],
            "ner_tags": labeled_df.groupby("id")["ner_tag"].apply(list).loc[filename],
            "bboxes": labeled_df.loc[labeled_df["id"] == filename, :][BBOX_COLS]
            .to_numpy()
            .tolist(),
            "image": image_dict[filename],
        }
        ner_annotations.append(annotation)

    return ner_annotations
