"""Test extraction tools/methods."""

import logging

import pandas as pd
import pytest
from dagster import Out, op
from mozilla_sec_eia.library.experiment_tracking import (
    get_most_recent_run,
    get_tracking_resource_name,
)
from mozilla_sec_eia.models.sec10k.extract import (
    ChunkFilingsConfig,
    compute_validation_metrics,
    extract_model_factory,
)
from mozilla_sec_eia.models.sec10k.utils.cloud import GCSArchive

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


def test_sec10k_extract_pipeline(
    filings_metadata,
    first_run_results,
    second_run_results,
    test_tracker_factory,
):
    """Test high level extraction workflow."""

    class FakeArchive(GCSArchive):
        filings_bucket_name: str = ""
        labels_bucket_name: str = ""
        metadata_db_instance_connection: str = ""
        user: str = ""
        metadata_db_name: str = ""
        project: str = ""

        def setup_for_execution(self, context):
            pass

        def get_metadata(self):
            return filings_metadata

    dataset_name = "test_pipeline"
    experiment_name = f"{dataset_name}_extraction"
    test_tracker = test_tracker_factory(experiment_name)

    for i, results in enumerate([first_run_results, second_run_results]):

        @op(out={"extraction_metadata": Out(), "extracted": Out()})
        def _fake_extract(_filings_to_extract):
            return results[0], results[1]

        test_job = extract_model_factory(dataset_name, _fake_extract)
        resources = {
            "basic_10k_extract_config": ChunkFilingsConfig(
                num_filings=3 if i == 0 else -1
            ),
            get_tracking_resource_name(experiment_name): test_tracker,
            "cloud_interface": FakeArchive(),
        }
        metadata = results[0]

        # Run extract method
        test_job.execute_in_process(resources=resources)
        run = get_most_recent_run(experiment_name, dagster_run_id="")
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
