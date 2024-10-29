"""Test for GAN train module."""

import os
from typing import Any
from unittest.mock import ANY, MagicMock, call, create_autospec, patch

import numpy as np
import pytest
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from src.enums import DeviceType
from src.gan.loss import DiscriminatorLoss
from src.gan.train import (
    evaluate,
    evaluate_and_checkpoint,
    initialize_training_state,
    log_generator_discriminator_metrics,
    train,
    train_disc,
    train_gen,
)
from src.gan.update_g import UpdateGenerator
from src.metrics.fid.fid import FID
from src.metrics.hubris import Hubris
from src.metrics.loss_term import LossSecondTerm
from src.models import (
    CheckpointGAN,
    ConfigGAN,
    DisParams,
    FIDMetricsParams,
    GANTrainArgs,
    GenParams,
    MetricsParams,
    TrainingState,
)
from src.utils.metrics_logger import MetricsLogger


@pytest.fixture
def fid_stats_file(tmp_path: str) -> None:
    """Fixture to create a temporary FID stats file."""
    stats_file = os.path.join(tmp_path, "fid_stats.npz")
    np.savez(
        stats_file,
        real_sum=np.random.rand(2048),
        real_cov_sum=np.random.rand(2048, 2048),
        num_real_images=np.array(1000),
    )
    return str(stats_file)


class MockFID(FID):
    """Custom FID mock class."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Init method for Mock FID."""
        super().__init__(*args, **kwargs)
        self.finalize = MagicMock()
        self.update = MagicMock()


class MockHubris(Hubris):
    """Mock Hubris metric."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init method for mock hubris."""
        super().__init__(*args, **kwargs)
        self.finalize = MagicMock()


class MockLossSecondTerm(LossSecondTerm):
    """Mock for loss second term."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init method for loss second therm mock."""
        super().__init__(*args, **kwargs)
        self.finalize = MagicMock()


class MockDataset(Dataset):
    """Mock dataset for training GAN."""

    def __init__(self, num_samples: int = 100, img_size: tuple[int, int, int] = (3, 32, 32)) -> None:
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            img_size (tuple): The size of each image in the dataset (C, H, W).

        """
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self) -> int:
        """Get number of samples in mock dataset."""
        return self.num_samples

    def __getitem__(self, _: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a random tensor with the specified size to simulate an image."""
        image = torch.randn(self.img_size)

        # Dummy label
        label = torch.randint(0, 10, (1,)).item()  # Assume 10 classes for classification tasks

        return image, label


@pytest.fixture
def mock_dataset() -> Dataset:
    """Fixture to create a mock dataset for testing."""
    dataset = MockDataset(num_samples=64, img_size=(3, 32, 32))  # 64 samples, 3-channel 32x32 images
    return dataset


@pytest.fixture
def gan_train_args(mock_dataset: Dataset, fid_stats_file: str) -> GANTrainArgs:
    """Fixture to create GANTrainArgs for testing."""
    # Create mocks for G and D
    G = create_autospec(nn.Module, instance=True)
    G.params = create_autospec(GenParams, instance=True)
    G.params.z_dim = 100

    # Set the `training` attribute and mock `eval` and `train` methods.
    G.training = True
    G.eval = MagicMock()
    G.train = MagicMock()

    # Mock the return value for G (generator)
    fake_generated_images = torch.randn(
        64, 3, 32, 32
    )  # Mocked generated images with batch size 64 and image size 32x32
    G.return_value = fake_generated_images

    D = create_autospec(nn.Module, instance=True)
    D.params = create_autospec(DisParams, instance=True)

    # Create mocks for optimizers and other components
    g_opt = create_autospec(optim.Optimizer, instance=True)
    d_opt = create_autospec(optim.Optimizer, instance=True)

    g_updater = create_autospec(UpdateGenerator, instance=True)
    g_updater.return_value = (
        torch.tensor(1.0, requires_grad=True),  # Mocked generator loss tensor
        {"term_1": 0.5, "term_2": 0.3},  # Mocked generator loss terms dictionary
    )
    g_updater.get_loss_terms.return_value = ["term_1", "term_2"]

    d_crit = create_autospec(DiscriminatorLoss, instance=True)

    # Configure `d_crit` to return a tensor for loss and a dictionary for loss terms
    d_loss_mock = torch.tensor(1.0, requires_grad=True)  # Mocked discriminator loss
    d_loss_terms_mock = {
        "term_1": 0.5,
        "term_2": 0.5,
    }
    d_crit.return_value = (d_loss_mock, d_loss_terms_mock)
    d_crit.get_loss_terms.return_value = ["term_1", "term_2"]

    # Set the test noise and metrics
    test_noise = torch.randn(64, 100)

    # Create instances of metrics
    feature_map_fn = MagicMock()
    feature_map_fn.return_value = torch.randn(64, 2048)  # Mock feature map output with expected dimensions

    # Create a mock `C` for the `Hubris` metric with a `get` method
    C_mock = MagicMock()
    mock_probs = torch.randn(64)
    mock_probs = torch.softmax(mock_probs, dim=0)  # Apply softmax to get valid probabilities
    C_mock.get = MagicMock(return_value=(mock_probs, [torch.randn(64) for _ in range(3)]))

    fid_metric = MockFID(
        feature_map_fn=feature_map_fn, dims=2048, n_images=64, device=DeviceType.cpu, fid_stats_file=fid_stats_file
    )
    hubris_metric = MockHubris(C=C_mock, dataset_size=64)
    loss_second_term_metric = MockLossSecondTerm(C=MagicMock())

    # Instantiate FIDMetricsParams with actual metrics
    fid_metrics = FIDMetricsParams(
        fid=fid_metric,
        focd=None,
        conf_dist=loss_second_term_metric,
        hubris=hubris_metric,
    )

    # Return the GANTrainArgs with the necessary components
    return GANTrainArgs(
        G=G,
        g_opt=g_opt,
        g_updater=g_updater,
        D=D,
        d_opt=d_opt,
        d_crit=d_crit,
        test_noise=test_noise,
        fid_metrics=fid_metrics,
        n_disc_iters=1,
        epochs=10,
        batch_size=64,
        lr=0.0005,
        device="cpu",  # Use "cpu" or "cuda" as a string as required by the Pydantic model
        early_stop=None,
        start_early_stop_when=None,
        checkpoint_dir="./checkpoints",
        checkpoint_every=5,
        fixed_noise=None,
        c_out_hist=None,
        classifier=None,
        out_dir="./output",  # Add the required `out_dir` field
        dataset=mock_dataset,
    )


@pytest.fixture
def config_gan() -> ConfigGAN:
    """Fixture to create ConfigGAN for testing."""
    dataset = {"name": "mnist", "binary": {"pos": 0, "neg": 1}}

    model = {
        "z_dim": 100,
        "architecture": {
            "name": "dcgan",
            "g_filter_dim": 64,
            "d_filter_dim": 64,
            "g_num_blocks": 3,
            "d_num_blocks": 3,
        },
        "loss": {"name": "wgan-gp", "args": 10.0},
    }

    optimizer = {"lr": 0.0005, "beta1": 0.5, "beta2": 0.999}

    train = {
        "step_1": {"epochs": 10, "checkpoint_every": 2, "batch_size": 64, "disc_iters": 5},
        "step_2": {
            "epochs": 5,
            "checkpoint_every": 1,
            "batch_size": 64,
            "disc_iters": 3,
            "classifier": ["path/to/classifier"],
            "weight": [{"kldiv": {"alpha": [0.5, 0.7]}}],
        },
    }

    return ConfigGAN(
        project="TestProject",
        name="TestRun",
        out_dir="./output",
        data_dir="./data",
        fid_stats_path="./fid_stats",
        fixed_noise=64,
        test_noise="./test_noise",
        compute_fid=True,
        device="cpu",
        num_workers=4,
        num_runs=1,
        step_1_seeds=[42],
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        train=train,
    )


@pytest.fixture
@patch("wandb.init")  # Mock wandb.init globally for this fixture
@patch("wandb.define_metric")  # Mock define_metric globally for this fixture
def metrics_gan(
    mock_wandb_init: MagicMock, mock_define_metric: MagicMock, gan_train_args: MagicMock
) -> tuple[MetricsLogger, MetricsLogger]:
    """Define GAN metrics for test."""
    eval_metrics = MetricsLogger(MetricsParams(prefix="validation", log_epoch=True))
    train_metrics = MetricsLogger(MetricsParams(prefix="train", log_epoch=True))
    log_generator_discriminator_metrics(train_metrics, eval_metrics, gan_train_args)
    train_metrics.add("term_1", iteration_metric=True)
    train_metrics.add("term_2", iteration_metric=True)

    return train_metrics, eval_metrics


def test_initialize_training_state(gan_train_args: MagicMock) -> None:
    """Test the initialize_training_state function."""
    training_state = initialize_training_state(params=gan_train_args)
    assert isinstance(training_state, TrainingState)
    assert training_state.epoch == 0
    assert training_state.early_stop_tracker == 0
    assert training_state.best_epoch == 0
    assert training_state.best_epoch_metric == float("inf")


@patch("wandb.init")  # Mock wandb.init to avoid initialization errors
@patch("wandb.define_metric")  # Mock define_metric to avoid initialization errors
def test_evaluate(
    mock_define_metric: MagicMock, mock_wandb_init: MagicMock, gan_train_args: MagicMock, metrics_gan: MagicMock
) -> None:
    """Test the evaluate function."""
    train_metrics, eval_metrics = metrics_gan
    evaluate(params=gan_train_args, stats_logger=eval_metrics)
    gan_train_args.G.eval.assert_called()  # Ensure eval mode was called on G
    for _, metric in gan_train_args.fid_metrics.model_dump().items():
        if metric is not None:
            metric.finalize.assert_called()  # Ensure metrics were finalized


@patch("wandb.init")  # Mock wandb.init to avoid initialization errors
@patch("wandb.define_metric")  # Mock define_metric to avoid initialization errors
def test_train_disc(
    mock_define_metric: MagicMock, mock_wandb_init: MagicMock, gan_train_args: MagicMock, metrics_gan: MagicMock
) -> None:
    """Test the train_disc function."""
    train_metrics, eval_metrics = metrics_gan
    real_data = torch.randn(gan_train_args.batch_size, 3, 32, 32)  # Mocked real data
    d_loss, d_loss_terms = train_disc(params=gan_train_args, train_metrics=train_metrics, real_data=real_data, fake_data=torch.randn(64, 100))
    assert isinstance(d_loss, torch.Tensor)
    assert isinstance(d_loss_terms, dict)
    gan_train_args.D.zero_grad.assert_called()  # Ensure gradients were zeroed
    gan_train_args.d_opt.step.assert_called()  # Ensure optimizer step was called


@patch("wandb.init")  # Mock wandb.init to avoid initialization errors
@patch("wandb.define_metric")  # Mock define_metric to avoid initialization errors
def test_train_gen(
    mock_define_metric: MagicMock, mock_wandb_init: MagicMock, gan_train_args: MagicMock, metrics_gan: MagicMock
) -> None:
    """Test the train_gen function."""
    train_metrics, eval_metrics = metrics_gan

    # Perform the generator training step
    g_loss, g_loss_terms = train_gen(params=gan_train_args, train_metrics=train_metrics, fake_data=torch.randn(64, 100))

    # Assert that g_loss is a tensor
    assert isinstance(g_loss, torch.Tensor), "Generator loss should be a tensor."
    assert isinstance(g_loss_terms, dict), "Generator loss terms should be returned as a dictionary."

    expected_call = call(gan_train_args.G, gan_train_args.D, gan_train_args.g_opt, ANY, gan_train_args.device)

    # Assert that the expected call was made
    assert (
        expected_call in gan_train_args.g_updater.call_args_list
    ), f"Expected call {expected_call} not found in actual call list: {gan_train_args.g_updater.call_args_list}"


@patch("wandb.init")  # Mock wandb.init to avoid initialization errors
@patch("wandb.define_metric")  # Mock define_metric to avoid initialization errors
@patch("src.gan.train.checkpoint_gan")
def test_evaluate_and_checkpoint(
    mock_checkpoint_gan: MagicMock,
    mock_define_metric: MagicMock,
    mock_wandb_init: MagicMock,
    gan_train_args: MagicMock,
    config_gan: MagicMock,
    metrics_gan: MagicMock,
) -> None:
    """Test the evaluate_and_checkpoint function."""
    # Set the mock to return a specific value, like a valid checkpoint path or None
    mock_checkpoint_gan.return_value = "mocked_checkpoint_path.pth"

    train_state = TrainingState(epoch=5)
    train_metrics, eval_metrics = metrics_gan
    train_metrics.finalize_epoch = MagicMock()
    eval_metrics.finalize_epoch = MagicMock()
    latest_cp = evaluate_and_checkpoint(
        params=gan_train_args,
        train_state=train_state,
        eval_metrics=eval_metrics,
        train_metrics=train_metrics,
        config=CheckpointGAN(config=config_gan, gen_params=gan_train_args.G.params, dis_params=gan_train_args.D.params),
    )
    gan_train_args.G.eval.assert_called()  # Ensure G was evaluated
    assert isinstance(latest_cp, str | type(None))  # Ensure latest_cp is a string or None


@patch("src.utils.metrics_logger.wandb.init")  # Mock wandb.init to avoid initialization errors
@patch("src.utils.metrics_logger.wandb.define_metric")  # Mock define_metric to avoid initialization errors
@patch("src.utils.metrics_logger.wandb.log")  # Mock define_metric to avoid initialization errors
@patch("src.gan.train.checkpoint_gan")  # Patch checkpoint_gan to prevent actual saving
def test_train(
    mock_checkpoint_gan: MagicMock,
    mock_define_log: MagicMock,
    mock_define_metric: MagicMock,
    mock_wandb_init: MagicMock,
    gan_train_args: MagicMock,
    config_gan: MagicMock,
) -> None:
    """Test the train function."""
    train_state, latest_cp, train_metrics, eval_metrics = train(gan_train_args, config_gan)
    assert isinstance(train_state, TrainingState)
    assert isinstance(latest_cp, MagicMock)
    assert isinstance(train_metrics, MetricsLogger)
    assert isinstance(eval_metrics, MetricsLogger)
    assert train_state.epoch == gan_train_args.epochs
