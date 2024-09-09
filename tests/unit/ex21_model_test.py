"""Unit tests for the LayoutLM model and table extractor."""

import torch

from mozilla_sec_eia.ex_21.inference import get_flattened_mode_predictions
from mozilla_sec_eia.ex_21.train_extractor import LABELS
from mozilla_sec_eia.utils.layoutlm import get_id_label_conversions


def test_bbox_overlap_prediction_tie_break():
    """Test the tie breaking process for overlapping bbox predictions.

    When handling multi page documents LayoutLM uses a sliding 'frame'
    with some overlap between frames. The overlap creates multiple
    predictions for the same bounding boxes. If there are multiple mode
    predictions for a bounding box, then ties are broken by setting
    a priority for the labels and choosing the highest priority label.
    """
    _, label2id = get_id_label_conversions(LABELS)
    example_boxes = torch.tensor(
        [
            [[0, 0, 0, 0], [0, 1, 1, 1], [1, 2, 3, 4], [2, 3, 4, 5]],
            [[1, 2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 6], [0, 0, 0, 0]],
        ]
    )
    predicted_labels = [
        "O",
        "B-Subsidiary",
        "I-Subsidiary",
        "B-Loc",
        "B-Subsidiary",
        "B-Loc",
        "I-Loc",
        "O",
    ]
    predicted_ids = torch.tensor(
        [label2id[label] for label in predicted_labels]
    ).unsqueeze(dim=1)
    expected_labels = [
        "O",
        "B-Subsidiary",
        "I-Subsidiary",
        "B-Loc",
        "I-Subsidiary",
        "B-Loc",
        "I-Loc",
        "O",
    ]
    mode_labels = get_flattened_mode_predictions(example_boxes, predicted_ids)
    assert ([label2id[label] for label in expected_labels] == mode_labels).all()
