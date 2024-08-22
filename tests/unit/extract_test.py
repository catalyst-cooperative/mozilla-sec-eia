"""Test extraction tools/methods."""

import logging
import unittest

import mlflow
import numpy as np
import pandas as pd
import pytest
from mozilla_sec_eia.extract import (
    _fill_nulls_for_comparison,
    _get_most_recent_run,
    clean_ex21_validation_set,
    compute_precision_and_recall,
    extract_filings,
    jaccard_similarity,
    strip_down_company_names,
)

logger = logging.getLogger(f"catalystcoop.{__name__}")


@pytest.fixture
def filings_metadata() -> pd.DataFrame:
    """Return fake filing metadata."""
    return pd.DataFrame(
        {
            "filename": [
                "filing1",
                "filing2",
                "filing3",
                "filing3",
                "filing4",
                "filing5",
            ],
        }
    )


@pytest.fixture
def first_run_results():
    """Metadata and extracted table from first run of extractor."""
    return (
        pd.DataFrame(
            {
                "filename": ["filing1", "filing2", "filing3"],
                "success": [True, True, False],
            }
        ).set_index("filename"),
        pd.DataFrame({"column": ["extracted table (not needed for test)"]}),
    )


@pytest.fixture
def second_run_results():
    """Metadata and extracted table from first run of extractor."""
    return (
        pd.DataFrame(
            {
                "filename": ["filing1", "filing2", "filing3", "filing4", "filing5"],
                "success": [True, True, False, True, True],
            },
        ).set_index("filename"),
        pd.DataFrame({"column": ["extracted table (not needed for test)"]}),
    )


def test_extract_basic_10k(
    test_mlflow_init_func,
    filings_metadata,
    first_run_results,
    second_run_results,
    tmp_path,
):
    """Test high level extraction workflow."""

    class FakeArchive:
        def get_metadata(self):
            return filings_metadata

    with (
        unittest.mock.patch("mozilla_sec_eia.extract.initialize_mlflow"),
        unittest.mock.patch("mozilla_sec_eia.extract.GCSArchive", new=FakeArchive),
    ):
        # Initialize mlflow with test settings
        test_mlflow_init_func()
        mlflow.create_experiment(
            "basic_10k_extraction", artifact_location=str(tmp_path)
        )

        for i, results in enumerate([first_run_results, second_run_results]):
            kwargs = {"num_filings": 3} if i == 0 else {"continue_run": True}
            with unittest.mock.patch(
                "mozilla_sec_eia.extract.basic_10k.extract", new=lambda *args: results
            ):
                metadata = results[0]

                # Run extract method
                extract_filings("basic_10k", **kwargs)
                run = _get_most_recent_run("basic_10k_extraction")
                assert run.data.metrics["num_failed"] == (~metadata["success"]).sum()
                assert run.data.metrics["ratio_extracted"] == len(metadata) / len(
                    filings_metadata
                )


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
def test_compute_precision_and_recall(
    computed_set, validation_set, expected_precision, expected_recall
):
    """Test validation metrics with test sets."""
    metrics = compute_precision_and_recall(computed_set, validation_set, "value")

    assert metrics["precision"] == expected_precision
    assert metrics["recall"] == expected_recall


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
        computed_df[col] = _fill_nulls_for_comparison(computed_df[col])
        validation_df[col] = _fill_nulls_for_comparison(validation_df[col])
    logger.info(f"validation: {validation_df}")
    logger.info(f"computed: {computed_df}")
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
