"""Module for testing histograms for wandb."""

from collections.abc import Generator
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import PIL
import pytest
import torch

from src.classifier.classifier_cache import ClassifierCache
from src.metrics.c_output_hist import OutputsHistogram
from src.metrics.hubris import Hubris


@pytest.fixture
def mock_classifier_cache() -> Generator[ClassifierCache, None, None]:
    """Fixture to provide a mock classifier cache."""
    mock_cache = MagicMock(spec=ClassifierCache)

    # Create a mock for the C attribute
    mock_C = Mock()
    mock_C.models = [MagicMock(), MagicMock()]
    mock_cache.configure_mock(C=mock_C)

    yield mock_cache


@pytest.fixture
def mock_hubris() -> Generator[Hubris, None, None]:
    """Fixture to provide a mock hubris object."""
    with patch("src.metrics.hubris.Hubris") as mock_hubris_class:
        mock_hubris_instance = mock_hubris_class.return_value
        mock_hubris_instance.update = MagicMock()
        mock_hubris_instance.get_clfs.return_value = [0.1, 0.2]
        mock_hubris_instance.finalize.return_value = 0.5
        yield mock_hubris_instance


@pytest.fixture
def outputs_histogram(mock_classifier_cache: ClassifierCache) -> OutputsHistogram:
    """Fixture to provide an instance of OutputsHistogram."""
    dataset_size = 100
    return OutputsHistogram(mock_classifier_cache, dataset_size)


def test_initialization(outputs_histogram: OutputsHistogram, mock_classifier_cache: ClassifierCache) -> None:
    """Test OutputsHistogram initialization."""
    assert outputs_histogram.C == mock_classifier_cache
    assert outputs_histogram.dataset_size == 100
    assert outputs_histogram.y_hat.shape == (100,)
    assert outputs_histogram.output_clfs == len(mock_classifier_cache.C.models)
    if outputs_histogram.output_clfs > 0:
        assert outputs_histogram.y_preds.shape == (outputs_histogram.output_clfs, 100)


@patch("src.metrics.c_output_hist.Hubris.update")
def test_update(mock_hubris_update: MagicMock, outputs_histogram: OutputsHistogram, mock_hubris: Hubris) -> None:
    """Test OutputsHistogram update with new batch of images."""
    # Mock data
    images = torch.randn((10, 3, 64, 64))  # Example batch of 10 images
    batch = (0, 10)  # Starting index and batch size

    # Mock behavior for C.get()
    mock_y_hat = torch.randn(10).float()
    mock_y_preds = [[torch.randn(10).float(), torch.randn(10).float()], mock_y_hat]
    outputs_histogram.C.get = MagicMock(return_value=(mock_y_hat, mock_y_preds))

    # Call the update function
    outputs_histogram.update(images, batch)

    # Assert hubris update is called
    mock_hubris_update.assert_called_once_with(images, batch)

    # Assert get function from ClassifierCache is called
    outputs_histogram.C.get.assert_called_once_with(images, batch[0], batch[1], output_feature_maps=True)

    # Check the updated y_hat values with close approximation
    assert torch.allclose(outputs_histogram.y_hat[0:10].float(), mock_y_hat, atol=1e-5), "y_hat values do not match."

    # Check the updated y_preds values with close approximation
    for i in range(outputs_histogram.output_clfs):
        assert torch.allclose(
            outputs_histogram.y_preds[i, 0:10].float(), mock_y_preds[0][i], atol=1e-5
        ), f"y_preds values for classifier {i} do not match."


def test_plot(outputs_histogram: OutputsHistogram) -> None:
    """Test OutputsHistogram plot function."""
    with patch("seaborn.histplot") as mock_histplot:
        outputs_histogram.plot()
        mock_histplot.assert_called_once_with(data=outputs_histogram.y_hat, stat="proportion", bins=20)


def test_plot_clfs(outputs_histogram: OutputsHistogram, mock_hubris: Hubris) -> None:
    """Test OutputsHistogram plot_clfs function."""
    outputs_histogram.output_clfs = 2
    outputs_histogram.y_hat = torch.rand((100,))
    outputs_histogram.y_preds = torch.rand((2, 100))

    placeholder_image = PIL.Image.new("RGB", (10, 10))
    with (
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("seaborn.kdeplot") as mock_kdeplot,
        patch("seaborn.scatterplot") as mock_scatterplot,
        patch("PIL.Image.frombytes", return_value=placeholder_image) as mock_image_frombytes,
        patch("matplotlib.pyplot.close"),
    ):

        mock_fig = MagicMock()
        mock_axs = MagicMock(spec=np.empty((2, 3)))
        mock_subplots.return_value = (mock_fig, mock_axs)

        # Mock methods for axs
        for ax in mock_axs.flatten():  # Flatten to easily iterate through all mock axes
            ax.set_xlim = MagicMock()
            ax.set_title = MagicMock()

        result = outputs_histogram.plot_clfs()

        assert result is not None
        assert mock_kdeplot.call_count == 5
        assert mock_scatterplot.call_count == 1
        mock_image_frombytes.assert_called_once()


def test_plot_clfs_no_clfs(outputs_histogram: OutputsHistogram) -> None:
    """Test OutputsHistogram plot_clfs function with no classifiers."""
    outputs_histogram.output_clfs = 0
    result = outputs_histogram.plot_clfs()
    assert result is None


def test_reset(outputs_histogram: OutputsHistogram) -> None:
    """Test OutputsHistogram reset method."""
    # The reset method currently does nothing, so we are only checking it doesn't throw errors
    outputs_histogram.reset()


def test_finalize(outputs_histogram: OutputsHistogram) -> None:
    """Test OutputsHistogram finalize method."""
    # The finalize method currently does nothing, so we are only checking it doesn't throw errors
    outputs_histogram.finalize()
