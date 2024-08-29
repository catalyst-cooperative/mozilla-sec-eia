"""Test extraction tools/methods."""

import logging

import pandas as pd
import pytest
from dagster import Out, RunConfig, op
from mozilla_sec_eia.library.experiment_tracking.mlflow_io_managers import (
    MlflowMetricsIOManager,
    MlflowPandasArtifactIOManager,
)
from mozilla_sec_eia.models.sec10k.extract import (
    FilingsToExtractConfig,
    extract_graph_factory,
)
from mozilla_sec_eia.models.sec10k.utils.cloud import GCSArchive

logger = logging.getLogger(f"catalystcoop.{__name__}")


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
        "mlflow_metrics_io_manager": MlflowMetricsIOManager(
            experiment_tracker=test_tracker,
        ),
    }
    graph_result = test_graph.to_job().execute_in_process(
        resources=resources,
        run_config=RunConfig(
            {"get_filings_to_extract": FilingsToExtractConfig(num_filings=num_filings)}
        ),
        input_values={
            "previous_extraction_metadata": previous_extraction_metadata,
            "previous_extracted": pd.DataFrame(),
        },
    )
    extraction_metadata, metrics = (
        graph_result.output_value("extraction_metadata"),
        graph_result.output_value("extraction_metrics"),
    )

    run = get_most_recent_mlflow_run_factory(experiment_name)
    assert run.data.metrics["num_failed"] == num_failed
    assert run.data.metrics["ratio_extracted"] == len(extraction_metadata) / len(
        filings_metadata
    )
    assert run.data.metrics == metrics
