"""Validate basic 10k and exhibit 21 extraction."""

from mozilla_sec_eia.extract import (
    _get_experiment_name,
    _get_most_recent_run,
    validate_extraction,
)


def test_basic_10k_extraction():
    """Run full 10k extraction on validation set and verify desired metrics are met."""
    validate_extraction("basic_10k")
    run = _get_most_recent_run(
        _get_experiment_name("basic_10k", experiment_suffix="validation")
    )

    assert run.data.metrics["precision"] == 1
    assert run.data.metrics["recall"] == 1
