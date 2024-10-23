"""Unit tests for the LayoutLM model and table extractor."""

import numpy as np
import pandas as pd
import pytest
import torch

from mozilla_sec_eia.library.validation_helpers import (
    fill_nulls_for_comparison,
    jaccard_similarity,
    pandas_compute_precision_recall,
    strip_down_company_names,
)
from mozilla_sec_eia.models.sec10k.ex_21.data.common import (
    LABELS,
    get_flattened_mode_predictions,
)
from mozilla_sec_eia.models.sec10k.ex_21.ex21_validation_helpers import (
    clean_ex21_validation_set,
)
from mozilla_sec_eia.models.sec10k.utils.layoutlm import get_id_label_conversions


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


@pytest.mark.parametrize(
    "computed_df,validation_df,expected_subsidiary_jaccard,expected_loc_jaccard,expected_own_per_jaccard",
    [
        (
            pd.DataFrame(
                {
                    "id": ["1", "1", "2", "3"],
                    "subsidiary": [
                        "atc management, inc.",
                        "american transmission company",
                        "bostco llc",
                        "acorpa",
                    ],
                    "loc": ["wisconsin", "wisconsin", "wisconsin", pd.NA],
                    "own_per": [26.5, 23.04, np.nan, np.nan],
                },
                index=[0, 1, 2, 3],
            ),
            pd.DataFrame(
                {
                    "Filename": ["1", "1", "2", "3"],
                    "Subsidiary": [
                        "atc management, inc",
                        "american transmission company llc",
                        "bostco llc ",
                        "acorpa",
                    ],
                    "Location of Incorporation": [
                        "wisconsin",
                        "wisconsin ",
                        " Wisconsin",
                        pd.NA,
                    ],
                    "Ownership Percentage": [26.5, 23.04, np.nan, np.nan],
                },
                index=[0, 1, 2, 3],
            ),
            1,
            1,
            1,
        ),
    ],
)
def test_ex21_validation_cleaning(
    computed_df,
    validation_df,
    expected_subsidiary_jaccard,
    expected_loc_jaccard,
    expected_own_per_jaccard,
):
    validation_df = clean_ex21_validation_set(validation_df=validation_df)
    computed_df["subsidiary"] = strip_down_company_names(computed_df["subsidiary"])
    validation_df["subsidiary"] = strip_down_company_names(validation_df["subsidiary"])
    for col in ["subsidiary", "loc", "own_per"]:
        computed_df[col] = fill_nulls_for_comparison(computed_df[col])
        validation_df[col] = fill_nulls_for_comparison(validation_df[col])
    assert expected_subsidiary_jaccard == jaccard_similarity(
        computed_df=computed_df, validation_df=validation_df, value_col="subsidiary"
    )
    assert expected_loc_jaccard == jaccard_similarity(
        computed_df=computed_df, validation_df=validation_df, value_col="loc"
    )
    assert expected_own_per_jaccard == jaccard_similarity(
        computed_df=computed_df, validation_df=validation_df, value_col="own_per"
    )
    return


@pytest.mark.parametrize(
    "computed_set,validation_set,expected_precision,expected_recall",
    [
        (
            pd.DataFrame({"value": ["a", "b", "c"]}, index=[0, 1, 2]),
            pd.DataFrame({"value": ["a", "b", "c"]}, index=[0, 1, 2]),
            1,
            1,
        ),
        (
            pd.DataFrame({"value": ["a", "b", "c", "d"]}, index=[0, 1, 2, 3]),
            pd.DataFrame({"value": ["a", "b", "c"]}, index=[0, 1, 2]),
            3 / 4,
            1,
        ),
        (
            pd.DataFrame({"value": ["a", "b", "c"]}, index=[0, 1, 2]),
            pd.DataFrame({"value": ["a", "b", "c", "d"]}, index=[0, 1, 2, 3]),
            1,
            3 / 4,
        ),
        (
            pd.DataFrame({"value": ["a", "b", "d"]}, index=[0, 1, 2]),
            pd.DataFrame({"value": ["a", "b", "c"]}, index=[0, 1, 2]),
            2 / 3,
            2 / 3,
        ),
        (
            pd.DataFrame(
                {"value": ["a", "b", "d"], "idx0": ["1", "2", "3"], "idx1": [4, 2, 1]}
            ).set_index(["idx0", "idx1"]),
            pd.DataFrame(
                {"value": ["a", "b", "c"], "idx0": ["1", "2", "3"], "idx1": [4, 2, 1]}
            ).set_index(["idx0", "idx1"]),
            2 / 3,
            2 / 3,
        ),
        (
            pd.DataFrame(
                {
                    "value": ["a", "b", "c", "d"],
                    "idx0": ["1", "2", "3", "4"],
                    "idx1": [4, 2, 1, 5],
                }
            ).set_index(["idx0", "idx1"]),
            pd.DataFrame(
                {"value": ["a", "b", "c"], "idx0": ["1", "2", "3"], "idx1": [4, 2, 1]}
            ).set_index(["idx0", "idx1"]),
            3 / 4,
            1,
        ),
        (
            pd.DataFrame(
                {"value": ["c", "b", "a"], "idx0": ["3", "2", "1"], "idx1": [1, 2, 4]}
            ).set_index(["idx0", "idx1"]),
            pd.DataFrame(
                {"value": ["a", "b", "c"], "idx0": ["1", "2", "3"], "idx1": [4, 2, 1]}
            ).set_index(["idx0", "idx1"]),
            1,
            1,
        ),
    ],
)
def test_pandas_compute_precision_and_recall(
    computed_set, validation_set, expected_precision, expected_recall
):
    """Test validation metrics with test sets."""
    metrics = pandas_compute_precision_recall(computed_set, validation_set, "value")

    assert metrics["precision"] == expected_precision
    assert metrics["recall"] == expected_recall
