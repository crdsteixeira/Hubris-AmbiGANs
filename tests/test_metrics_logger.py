"""Module for testing of metrics logger for wandb."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import wandb
from src.enums import TrainingStage
from src.models import MetricsParams
from src.utils.metrics_logger import MetricsLogger


@pytest.fixture
@patch("wandb.init")
def mock_metrics_params(mock_wandb_init: MagicMock) -> MetricsParams:
    """Fixture to provide mock MetricsParams and initialize WandB."""
    mock_wandb_init.return_value = None  # Mock wandb.init() to do nothing
    return MetricsParams(prefix=TrainingStage.test, log_epoch=True)


@pytest.fixture
@patch("wandb.define_metric")
@patch("wandb.log")
@patch("wandb.init")
def mock_logger(
    mock_wandb_init: MagicMock,
    mock_wandb_log: MagicMock,
    mock_wandb_define_metric: MagicMock,
    mock_metrics_params: MagicMock,
) -> MetricsLogger:
    """Fixture to provide a mock MetricsLogger instance."""
    mock_wandb_init.return_value = None
    return MetricsLogger(params=mock_metrics_params)


@patch("wandb.init")
@patch("wandb.Image")
@patch("wandb.define_metric")
def test_add_media_metric(
    mock_wandb_init: MagicMock, mock_wandb_image: MagicMock, mock_wandb_define_metric: MagicMock, mock_logger: MagicMock
) -> None:
    """Test adding a media metric and logging an image."""
    # Initialize WandB to prevent preinit errors
    mock_wandb_init.return_value = None
    wandb.init()

    # Call the methods to add a media metric and log an image
    mock_logger.add_media_metric("sample_media")
    mock_logger.log_image("sample_media", np.random.rand(64, 64, 3), caption="Sample Caption")

    # Assertions to verify behavior
    mock_wandb_image.assert_called_once()
    assert "test/sample_media" in mock_logger.log_dict
    assert mock_logger.log_dict["test/sample_media"] is not None


@patch("wandb.init")
@patch("wandb.define_metric")
def test_add(mock_wandb_define_metric: MagicMock, mock_wandb_init: MagicMock, mock_logger: MagicMock) -> None:
    """Test adding a metric to the logger."""
    mock_wandb_init.return_value = None
    wandb.init()

    # Call the method to add a metric
    mock_logger.add("test_metric")

    # Verify the correct call to wandb.define_metric
    mock_wandb_define_metric.assert_any_call("test/test_metric", step_metric="test/epoch")
    assert "test/test_metric" in mock_logger.log_dict
    assert "test_metric" in mock_logger.stats


@patch("wandb.init")
@patch("wandb.define_metric")
def test_add_iteration_metric(
    mock_wandb_define_metric: MagicMock, mock_wandb_init: MagicMock, mock_logger: MagicMock
) -> None:
    """Test adding an iteration metric to the logger."""
    mock_logger.add("iteration_metric", iteration_metric=True)
    assert "iteration_metric" in mock_logger.iteration_metrics
    assert "iteration_metric_per_it" in mock_logger.stats
    assert "iteration_metric" in mock_logger.running_stats
    assert "iteration_metric" in mock_logger.it_counter


@patch("wandb.init")
@patch("wandb.define_metric")
def test_update_iteration_metric(
    mock_wandb_define_metric: MagicMock, mock_wandb_init: MagicMock, mock_logger: MagicMock
) -> None:
    """Test updating an iteration metric value."""
    mock_logger.add("iteration_metric", iteration_metric=True)
    mock_logger.update_it_metric("iteration_metric", 5)
    assert mock_logger.running_stats["iteration_metric"] == 5
    assert mock_logger.it_counter["iteration_metric"] == 1

    mock_logger.update_it_metric("iteration_metric", 10)
    assert mock_logger.running_stats["iteration_metric"] == 15
    assert mock_logger.it_counter["iteration_metric"] == 2


@patch("wandb.init")
@patch("wandb.define_metric")
def test_update_epoch_metric(
    mock_wandb_define_metric: MagicMock, mock_wandb_init: MagicMock, mock_logger: MagicMock
) -> None:
    """Test updating an epoch metric value."""
    mock_logger.add("epoch_metric")
    mock_logger.update_epoch_metric("epoch_metric", 0.85)
    assert mock_logger.stats["epoch_metric"] == [0.85]
    assert mock_logger.log_dict["test/epoch_metric"] == 0.85


@patch("wandb.init")
@patch("wandb.define_metric")
def test_reset_iteration_metrics(
    mock_wandb_define_metric: MagicMock, mock_wandb_init: MagicMock, mock_logger: MagicMock
) -> None:
    """Test resetting iteration metrics."""
    mock_logger.add("iteration_metric", iteration_metric=True)
    mock_logger.update_it_metric("iteration_metric", 5)
    assert mock_logger.running_stats["iteration_metric"] == 5

    mock_logger.reset_it_metrics()
    assert mock_logger.running_stats["iteration_metric"] == 0
    assert mock_logger.it_counter["iteration_metric"] == 0


@patch("wandb.init")
@patch("wandb.define_metric")
@patch("wandb.log")
def test_finalize_epoch(
    mock_wandb_log: MagicMock, mock_wandb_define_metric: MagicMock, mock_wandb_init: MagicMock, mock_logger: MagicMock
) -> None:
    """Test finalizing the epoch."""
    mock_logger.add("epoch_metric")
    mock_logger.update_epoch_metric("epoch_metric", 0.95)

    mock_logger.add("iteration_metric", iteration_metric=True)
    mock_logger.update_it_metric("iteration_metric", 5)
    mock_logger.update_it_metric("iteration_metric", 15)

    mock_logger.finalize_epoch()

    # Check if the iteration metric is correctly averaged and logged
    expected_avg = (5 + 15) / 2
    assert mock_logger.stats["iteration_metric"] == [expected_avg]
    assert mock_logger.log_dict["test/iteration_metric"] == expected_avg

    # Ensure the epoch value was logged and incremented
    assert mock_logger.log_dict["test/epoch"] == 1
    assert mock_logger.epoch == 2

    # Verify that wandb.log was called
    mock_wandb_log.assert_called_once_with(mock_logger.log_dict, commit=True)


@patch("wandb.log")
def test_log_plot(mock_wandb_log: MagicMock, mock_logger: MagicMock) -> None:
    """Test logging a plot."""
    plt.plot([0, 1, 2], [0, 1, 4])
    mock_logger.log_plot("sample_plot")

    mock_wandb_log.assert_called_once_with({"test/sample_plot": plt})


@pytest.mark.parametrize(
    "prefix, name, expected",
    [
        ("test", "metric_name", "test/metric_name"),
        (None, "metric_name", "metric_name"),
    ],
)
def test_apply_prefix(mock_metrics_params: MagicMock, prefix: str, name: str, expected: str) -> None:
    """Test applying a prefix to a metric name."""
    mock_metrics_params.prefix = prefix
    logger = MetricsLogger(params=mock_metrics_params)
    result = logger.apply_prefix(name)
    assert result == expected
