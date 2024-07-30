"""Test extraction tools/methods."""

import unittest

import mlflow
import pandas as pd
import pytest
from mozilla_sec_eia.extract import (
    _get_most_recent_run,
    compute_validation_metrics,
    extract_filings,
)


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
def test_compute_validation_metrics(
    computed_set, validation_set, expected_precision, expected_recall
):
    """Test validation metrics with test sets."""
    metrics = compute_validation_metrics(computed_set, validation_set, "value")

    assert metrics["precision"] == expected_precision
    assert metrics["recall"] == expected_recall
