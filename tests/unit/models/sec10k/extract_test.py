"""Test extraction tools/methods."""

import logging

import pandas as pd
import pytest
from dagster import Out, RunConfig, op
from mozilla_sec_eia.library.experiment_tracking.mlflow_io_managers import (
    MlflowPandasArtifactIOManager,
)
from mozilla_sec_eia.models.sec10k.extract import (
    FilingsToExtractConfig,
    compute_validation_metrics,
    extract_graph_factory,
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


@pytest.mark.parametrize(
    "filings_metadata,previous_extraction_metadata,num_filings,num_failed",
    [
        (
            pd.DataFrame(
                {"filename": ["filing1", "filing2", "filing3", "filing4", "filing5"]}
            ),
            pd.DataFrame(
                {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
            ).set_index("filename"),
            -1,
            0,
        ),
        (
            pd.DataFrame(
                {"filename": ["filing1", "filing2", "filing3", "filing4", "filing5"]}
            ),
            pd.DataFrame(
                {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
            ).set_index("filename"),
            -1,
            3,
        ),
        (
            pd.DataFrame(
                {"filename": ["filing1", "filing2", "filing3", "filing4", "filing5"]}
            ),
            pd.DataFrame(
                {"filename": ["filing1", "filing2"], "success": [True, True]}
            ).set_index("filename"),
            -1,
            0,
        ),
        (
            pd.DataFrame(
                {"filename": ["filing1", "filing2", "filing3", "filing4", "filing5"]}
            ),
            pd.DataFrame(
                {"filename": pd.Series(dtype=str), "success": pd.Series(dtype=bool)}
            ).set_index("filename"),
            2,
            1,
        ),
    ],
)
def test_sec10k_extract_pipeline(
    filings_metadata,
    previous_extraction_metadata,
    num_filings,
    num_failed,
    test_tracker_factory,
    get_most_recent_mlflow_run_factory,
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

    @op(out={"extraction_metadata": Out(), "extracted": Out()})
    def test_extract(
        cloud_interface: GCSArchive,
        filings_to_extract: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        md = filings_to_extract
        md["success"] = True
        md.iloc[:num_failed, 1] = False
        return md.set_index("filename"), pd.DataFrame()

    dataset_name = "test_pipeline"
    experiment_name = f"{dataset_name}_extraction"
    test_tracker = test_tracker_factory(experiment_name)

    test_graph = extract_graph_factory("test_extract", test_extract)
    resources = {
        "experiment_tracker": test_tracker,
        "cloud_interface": FakeArchive(),
        "mlflow_pandas_artifact_io_manager": MlflowPandasArtifactIOManager(
            experiment_tracker=test_tracker
        ),
    }
    extraction_metadata = (
        test_graph.to_job()
        .execute_in_process(
            resources=resources,
            run_config=RunConfig(
                {
                    "get_filings_to_extract": FilingsToExtractConfig(
                        num_filings=num_filings
                    )
                }
            ),
            input_values={
                "previous_extraction_metadata": previous_extraction_metadata,
                "previous_extracted": pd.DataFrame(),
            },
        )
        .output_value()
    )

    run = get_most_recent_mlflow_run_factory(experiment_name)
    assert run.data.metrics["num_failed"] == num_failed
    assert run.data.metrics["ratio_extracted"] == len(extraction_metadata) / len(
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
