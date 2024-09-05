"""Util functions for extracting text from PDFs.

Used to prepare inputs for Ex. 21 extraction modeling.
Functions include drawing bounding boxes around words.
"""

import logging
from typing import Any

import cv2
import fitz
import numpy as np
import pandas as pd
from fitz import Rect
from matplotlib import pyplot as plt
from PIL import Image

PDF_POINTS_PER_INCH = 72  # believe this is standard for all PDFs
logger = logging.getLogger(__name__)


def get_pdf_data_from_path(pdf_path):
    """Get words, images, and bounding boxes from a PDF.

    Arguments:
        pdf_path: path to the PDF.

    Returns:
        A page metadata dictionary with keys "pdf_text", "image", and "page"
        and a PyMuPDF page object which represents all pages in the PDF.
    """
    # TODO: replace asserts within logging messages?
    assert pdf_path.exists()
    doc = fitz.Document(str(pdf_path))
    assert doc.is_pdf
    # keep this if statement so that one page docs don't change from v0
    pg = combine_doc_pages(doc) if len(doc) > 1 else doc[0]
    extracted = extract_pdf_data_from_page(pg)
    return extracted, pg


def extract_pdf_data_from_page(page: fitz.Page) -> dict[str, pd.DataFrame]:
    """Parse PDF page data."""
    contents = _parse_page_contents(page)
    meta = {
        "rotation_degrees": [page.rotation],
        "origin_x_pdf_coord": [page.rect[0]],
        "origin_y_pdf_coord": [page.rect[1]],
        "width_pdf_coord": [page.rect[2] - page.rect[0]],
        "height_pdf_coord": [page.rect[3] - page.rect[1]],
        "has_images": [not contents["image"].empty],
        "has_text": [not contents["pdf_text"].empty],
        "page_num": [page.number],
    }
    if not contents["image"].empty:
        img_area = (
            contents["image"]
            .eval(
                "((bottom_right_x_pdf - top_left_x_pdf)"
                " * (bottom_right_y_pdf - top_left_y_pdf))"
            )
            .sum()
        )
    else:
        img_area = 0
    total_area = meta["width_pdf_coord"][0] * meta["height_pdf_coord"][0]

    meta["image_area_frac"] = [np.float32(img_area / total_area)]
    meta_df = pd.DataFrame(meta).astype(
        {
            "rotation_degrees": np.int16,
            "origin_x_pdf_coord": np.float32,
            "origin_y_pdf_coord": np.float32,
            "width_pdf_coord": np.float32,
            "height_pdf_coord": np.float32,
            "has_images": "boolean",
            "has_text": "boolean",
            "page_num": np.int16,
            "image_area_frac": np.float32,
        }
    )
    meta = {"page": meta_df}
    for df in contents.values():  # add ID fields
        if not df.empty:
            df["page_num"] = np.int16(page.number)
    return contents | meta


def combine_doc_pages(doc):
    """Combine all pages in a PyMuPDF doc into one PyMuPDF page.

    Arguments:
        doc: The Fitz Document object with the multi page Ex. 21.

    Returns:
        A Fitz page with all pages of the Ex. 21 combined
        into one page.
    """
    combined_width = 0
    combined_height = 0
    rects = []
    blank_page_nums = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        extracted = extract_pdf_data_from_page(page)
        if extracted["pdf_text"].empty:
            blank_page_nums.append(page_num)
            rects.append(Rect())
            continue
        pg_width = page.rect.width
        combined_width = max(combined_width, pg_width)
        full_pg_height = page.rect.height
        # instead of using page height directly, use the height of the last word + a buffer
        pg_txt_height = extracted["pdf_text"].bottom_right_y_pdf.max() + (
            full_pg_height / 100
        )
        # Translate this page down by the height of the previous page
        rects.append(
            Rect(0, combined_height, pg_width, combined_height + pg_txt_height)
        )
        page.set_cropbox(Rect(0, 0, pg_width, pg_txt_height))
        combined_height += pg_txt_height

    output_pdf = fitz.open()
    combined_page = output_pdf.new_page(
        width=float(combined_width), height=float(combined_height)
    )

    for i in range(len(doc)):
        if i in blank_page_nums:
            continue
        combined_page.show_pdf_page(rects[i], doc, i)
    return combined_page


def _parse_page_contents(page: fitz.Page) -> dict[str, pd.DataFrame]:
    """Parse page contents using fitz.TextPage."""
    flags = fitz.TEXTFLAGS_DICT
    # try getting only words
    textpage = page.get_textpage(flags=flags)
    content = textpage.extractDICT()
    delim_str = f"{chr(8212)}{chr(151)}"
    words = textpage.extractWORDS(delimiters=delim_str)
    images = []
    text = []
    for block in content["blocks"]:
        if block["type"] == 0:
            # skip over text, we'll parse it by word blocks
            continue
        if block["type"] == 1:
            images.append(_parse_image_block(block))
        else:
            raise ValueError(f"Unknown block type: {block['type']}")
    for word_block in words:
        parsed = _parse_word_block(word_block)
        if not parsed.empty:
            text.append(parsed)
    text = pd.concat(text, axis=0, ignore_index=True) if text else pd.DataFrame()
    if images:
        images = pd.concat(
            (pd.DataFrame(image) for image in images), axis=0, ignore_index=True
        )
    else:
        images = pd.DataFrame()

    return {"pdf_text": text, "image": images}


def _parse_image_block(img_block: dict[str, Any]) -> pd.DataFrame:
    """Parse an image block from a fitz.TextPage.extractDICT() output."""
    top_left_x_pdf, top_left_y_pdf, bottom_right_x_pdf, bottom_right_y_pdf = img_block[
        "bbox"
    ]
    dpi = min(
        img_block["xres"], img_block["yres"]
    )  # should be equal; min() just in case
    out = pd.DataFrame(
        {
            "img_num": [img_block["number"]],
            "dpi": [dpi],
            "top_left_x_pdf": [top_left_x_pdf],
            "top_left_y_pdf": [top_left_y_pdf],
            "bottom_right_x_pdf": [bottom_right_x_pdf],
            "bottom_right_y_pdf": [bottom_right_y_pdf],
        }
    ).astype(
        {
            "img_num": np.int16,
            "dpi": np.int16,
            "top_left_x_pdf": np.float32,
            "top_left_y_pdf": np.float32,
            "bottom_right_x_pdf": np.float32,
            "bottom_right_y_pdf": np.float32,
        }
    )
    return out


def _parse_word_block(word_block: tuple) -> pd.DataFrame:
    """Parse a word block from a fitz.TextPage.extractWORDS() output."""
    out = {
        "top_left_x_pdf": [word_block[0]],
        "top_left_y_pdf": [word_block[1]],
        "bottom_right_x_pdf": [word_block[2]],
        "bottom_right_y_pdf": [word_block[3]],
        "text": [word_block[4]],
        "block_num": [word_block[5]],
        "line_num": [word_block[6]],
        "word_num": [word_block[7]],
    }
    out = pd.DataFrame(out).astype(
        {
            "block_num": np.int16,
            "line_num": np.int16,
            "word_num": np.int16,
            "text": "string",
            "top_left_x_pdf": np.float32,
            "top_left_y_pdf": np.float32,
            "bottom_right_x_pdf": np.float32,
            "bottom_right_y_pdf": np.float32,
        }
    )
    return out


def _frac_normal_ascii(text: str | bytes) -> float:
    """Fraction of characters that are normal ASCII characters."""
    # normal characters, from space to tilde, plus whitespace
    # see https://www.asciitable.com/
    sum_ = 0
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    for char in text:
        if (32 <= ord(char) <= 126) or char in "\t\n":
            sum_ += 1
    return sum_ / len(text)


def pil_to_cv2(image: Image.Image) -> np.ndarray:  # noqa: C901
    """Convert a PIL Image to an OpenCV image (numpy array)."""
    # copied from https://gist.github.com/panzi/1ceac1cb30bb6b3450aa5227c02eedd3
    # This covers the common modes, is not exhaustive.
    mode = image.mode
    new_image: np.ndarray
    if mode == "1":
        new_image = np.array(image, dtype=np.uint8)
        new_image *= 255
    elif mode == "L":
        new_image = np.array(image, dtype=np.uint8)
    elif mode == "LA" or mode == "La":
        new_image = np.array(image.convert("RGBA"), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == "RGB":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == "RGBA":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == "LAB":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_LAB2BGR)
    elif mode == "HSV":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
    elif mode == "YCbCr":
        # XXX: not sure if YCbCr == YCrCb
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_YCrCb2BGR)
    elif mode == "P" or mode == "CMYK":
        new_image = np.array(image.convert("RGB"), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == "PA" or mode == "Pa":
        new_image = np.array(image.convert("RGBA"), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"unhandled image color mode: {mode}")

    return new_image


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Create PIL Image from numpy pixel array."""
    if len(img.shape) == 2:  # single channel, AKA grayscale
        return Image.fromarray(img)
    # only handle BGR for now
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def display_img_array(img: np.ndarray, figsize=(5, 5), **kwargs):
    """Plot image array for jupyter sessions."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 2:  # grayscale
        return plt.imshow(img, cmap="gray", vmin=0, vmax=255, **kwargs)
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), **kwargs)


def overlay_bboxes(
    img: np.ndarray, bboxes: np.ndarray, color=(255, 0, 0)
) -> np.ndarray:
    """Overlay bounding boxes of shape N x 4 (x0, y0, x1, y1) on an image."""
    img = img.copy()
    for box in np.round(bboxes, 0).astype(np.int32):  # float to int just in case:
        x0, y0, x1, y1 = box
        cv2.rectangle(img, (x0, y0), (x1, y1), color=color, thickness=1)
    return img


def pdf_coords_to_pixel_coords(coords: np.ndarray, dpi: int) -> np.ndarray:
    """Convert PDF coordinates to pixel coordinates."""
    # For arbitrary PDFs you would need to subtract the origin in PDF coordinates,
    # but since you create these PDFs, you know the origin is (0, 0).
    out = coords * dpi / PDF_POINTS_PER_INCH
    return out


def get_label_studio_bbox_dicts(
    bbox: pd.Series, ind, x_norm, y_norm, original_width, original_height
) -> list[dict]:
    """Create a dictionary to label a bbox in Label Studio.

    A Label Studio task dictionary is made up of these dictionaries
    for each of the word bounding boxes in a PDF. The full task
    dictionary can be used as an input to label the doc in Label Studio.

    Arguments:
        bbox: The bounding box for the word
        ind: The index of the word/bounding box in the full PDF
            dataframe.
        x_norm: The value by which to normalize the x coordinates.
        y_norm: The value by which to normalize the y coordinates.
        original_width: The original width of the PDF.
        original_height: The original height of the PDF.
    """
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


def render_page(pg: fitz.Page, dpi=150, clip: fitz.Rect | None = None) -> Image.Image:
    """Render a page of a PDF as a PIL.Image object.

    Args:
        pg (fitz.Page): a page of a PDF
        dpi (int, optional): image resolution in pixels per inch. Defaults to 150.
        clip (Optional[fitz.Rect], optional): Optionally render only a subset of the
            page. Defined in PDF coordinates. Defaults to None, which renders the
            full page.

    Returns:
        Image.Image: PDF page rendered as a PIL.Image object
    """
    # 300 dpi is what tesseract recommends. PaddleOCR seems to do fine with half that.
    render: fitz.Pixmap = pg.get_pixmap(dpi=dpi, clip=clip)
    img = _pil_img_from_pixmap(render)
    return img


def _pil_img_from_pixmap(pix: fitz.Pixmap) -> Image.Image:
    """Convert pyMuPDF Pixmap object to PIL.Image object.

    For some reason pyMuPDF (aka fitz) lets you save images using PIL, but does not
    have any function to convert to PIL objects. Clearly they do this conversion
    internally; they should just expose it. Instead, I had to copy it out from their
    source code.

    Args:
        pix (fitz.Pixmap): a rendered Pixmap

    Returns:
        Image: a PIL.Image object
    """
    # pyMuPDF source code on GitHub is all in SWIG (some kind of C to python code
    # generator) and is unreadable to me. So you have to inspect your local .py files.
    # Adapted from the Pixmap.pil_save method in python3.9/site-packages/fitz/fitz.py
    # I just replaced instances of "self" with "pix"
    cspace = pix.colorspace
    if cspace is None:
        mode = "L"
    elif cspace.n == 1:
        mode = "L" if pix.alpha == 0 else "LA"
    elif cspace.n == 3:
        mode = "RGB" if pix.alpha == 0 else "RGBA"
    else:
        mode = "CMYK"

    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img
