"""Module for testing GAN CLI."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from src.gan.gan_cli import (
    main,
    parse_args,
    train_modified_gan,
    train_step1_gan,
    train_step2_gan,
)
from src.gan.loss import DiscriminatorLoss
from src.gan.update_g import UpdateGenerator, UpdateGeneratorGAN
from src.metrics.fid.fid import FID
from src.metrics.focd import FOCD
from src.models import (
    CLAmbigan,
    ConfigCD,
    ConfigDatasetParams,
    ConfigGAN,
    ConfigLoss,
    ConfigModel,
    ConfigOptimizer,
    ConfigStep1,
    ConfigStep2,
    ConfigTrain,
    ConfigWeights,
    DisParams,
    FIDMetricsParams,
    GenParams,
    ImageParams,
    MetricsParams,
    Step1TrainingArgs,
    Step2TrainingArgs,
    TrainingStage,
)
from src.utils.metrics_logger import MetricsLogger


@pytest.fixture
def mock_config() -> ConfigGAN:
    """Fixture to provide mock configuration for ConfigGAN."""
    dataset = ConfigDatasetParams(name="mnist", binary={"pos": 1, "neg": 0})

    model = ConfigModel(
        z_dim=100,
        architecture={"name": "dcgan", "g_filter_dim": 64, "d_filter_dim": 64, "g_num_blocks": 4, "d_num_blocks": 4},
        loss=ConfigLoss(name="ns"),
    )

    optimizer = ConfigOptimizer(lr=0.001, beta1=0.5, beta2=0.999)

    train = ConfigTrain(
        step_1=ConfigStep1(epochs=10, batch_size=32, checkpoint_every=5, disc_iters=1),
        step_2=ConfigStep2(
            epochs=10,
            batch_size=32,
            checkpoint_every=5,
            disc_iters=1,
            classifier=["test_classifier"],
            weight=[ConfigWeights(cd=ConfigCD(alpha=[0.5]))],
        ),
    )

    return ConfigGAN(
        project="test_project",
        name="test_run",
        out_dir="out",
        data_dir="data",
        fid_stats_path="fid_path",
        fixed_noise=100,
        test_noise="test_noise_path",
        compute_fid=True,
        device="cpu",
        num_workers=4,
        num_runs=1,
        step_1_seeds=[42],
        step_2_seeds=[43],
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        train=train,
    )


@pytest.fixture
def mock_step1_args(mock_config: ConfigGAN) -> Step1TrainingArgs:
    """Fixture to provide mock Step1TrainingArgs."""
    image_params = ImageParams(image_size=(1, 6, 6))
    dataset = FakeData(size=100, image_size=(1, 6, 6), num_classes=10, transform=ToTensor())

    # Mocked Generator (G)
    G = MagicMock(spec=nn.Module)
    G.params = GenParams(image_size=(1, 6, 6), z_dim=1, n_blocks=2, filter_dim=10)

    # Mocked Discriminator (D)
    D = MagicMock(spec=nn.Module)
    D.params = DisParams(image_size=(1, 6, 6), filter_dim=10, n_blocks=2)

    g_updater = UpdateGeneratorGAN(crit=MagicMock())  # You can use a mock loss here.

    return Step1TrainingArgs(
        img_size=image_params,
        run_id=1,
        test_noise_conf={"mock": "conf"},
        fid_metrics=FIDMetricsParams(fid=MagicMock(spec=FID)),  # Mock fid metrics to keep it simple
        seed=42,
        checkpoint_dir="mock_dir",
        device=mock_config.device,
        dataset=dataset,
        fixed_noise=torch.randn(10, 100, 1, 1),  # Random noise tensor to match ConvTranspose2d input
        test_noise=torch.randn(10, 100, 1, 1),  # Random noise tensor
        G=G,
        g_opt=MagicMock(spec=optim.Optimizer),  # Mock optimizer
        g_updater=g_updater,
        D=D,
        d_opt=MagicMock(spec=optim.Optimizer),  # Mock optimizer
        d_crit=MagicMock(spec=DiscriminatorLoss),  # Mock the Discriminator criterion
        n_disc_iters=1,  # Number of discriminator iterations per generator iteration
        epochs=10,
        out_dir="mock_out_dir",
    )


@pytest.fixture
def mock_step2_args(mock_step1_args) -> Step2TrainingArgs:
    """Fixture to provide mock Step2TrainingArgs."""
    return Step2TrainingArgs(
        run_id=1234,
        seed=43,
        checkpoint_dir="mock_dir",
        dataset=mock_step1_args.dataset,
        fixed_noise=mock_step1_args.fixed_noise,
        test_noise=mock_step1_args.test_noise,
        s1_epoch="1",  # Should be a string for Pydantic
        gan_path="gan_path",
        c_name="classifier_name",
        weight=("weight_name", mock_step1_args.g_updater),
        G=mock_step1_args.G,
        g_opt=mock_step1_args.g_opt,
        g_updater=mock_step1_args.g_updater,
        D=mock_step1_args.D,
        d_opt=mock_step1_args.d_opt,
        d_crit=mock_step1_args.d_crit,
        fid_metrics=mock_step1_args.fid_metrics,
        n_disc_iters=1,
        epochs=10,
        out_dir="out_dir",
        device="cpu",
    )


@patch("argparse.ArgumentParser.parse_args")
def test_parse_args(mock_parse_args: MagicMock) -> None:
    """Test parsing arguments with Pydantic validation."""
    mock_parse_args.return_value = MagicMock(config_path="mock_config.yaml")
    with patch("src.gan.gan_cli.CLAmbigan.model_validate") as mock_validate:
        mock_validate.return_value = CLAmbigan(config_path="mock_config.yaml")
        args: CLAmbigan = parse_args()
        assert args.config_path == "mock_config.yaml"
        mock_validate.assert_called_once()


@patch("src.gan.gan_cli.construct_gan")
@patch("src.gan.gan_cli.construct_optimizers")
@patch("torch.save")
@patch("src.gan.gan_cli.train")
def test_train_step1_gan(
    mock_train: MagicMock,
    mock_torch_save: MagicMock,
    mock_construct_optimizers: MagicMock,
    mock_construct_gan: MagicMock,
    mock_config: ConfigGAN,
    mock_step1_args: Step1TrainingArgs,
) -> None:
    """Test training GAN step 1 initialization."""
    # Mock the construct_gan return value with mocked G and D
    mock_construct_gan.return_value = (mock_step1_args.G, mock_step1_args.D)

    # Mock the construct_optimizers return value with mock optimizers
    mock_construct_optimizers.return_value = (mock_step1_args.g_opt, mock_step1_args.d_opt)
    mock_train.return_value = ({}, None, None, None)

    step_1_train_state, checkpoint_dir = train_step1_gan(mock_step1_args, mock_config)

    mock_construct_gan.assert_called_once()
    mock_construct_optimizers.assert_called_once()
    assert step_1_train_state is not None
    assert checkpoint_dir is not None


@patch("src.gan.gan_cli.construct_weights")
@patch("src.gan.gan_cli.construct_gan_from_checkpoint")
@patch("src.gan.gan_cli.construct_classifier_from_checkpoint")
@patch("src.gan.gan_cli.construct_optimizers")
@patch("torch.save")
@patch("src.gan.gan_cli.train")
@patch("src.gan.gan_cli.FOCD")
@patch("src.gan.gan_cli.plot_metrics")
def test_train_step2_gan(
    mock_plot_metrics: MagicMock,
    mock_focd: MagicMock,
    mock_train: MagicMock,
    mock_torch_save: MagicMock,
    mock_construct_optimizers: MagicMock,
    mock_construct_classifier: MagicMock,
    mock_construct_gan_from_checkpoint: MagicMock,
    mock_construct_weights: MagicMock,
    mock_step2_args: Step2TrainingArgs,
    mock_config: ConfigGAN,
) -> None:
    """Test training GAN step 2 with classifiers."""
    mock_classifier = MagicMock(spec=nn.Module)
    mock_stats = MetricsLogger(params=MetricsParams(prefix=TrainingStage.test))
    mock_stats.stats["fid"] = [13]
    mock_stats.stats["hubris"] = [0.50]
    mock_stats.stats["conf_dist"] = [0.10]
    mock_construct_classifier.return_value = (mock_classifier, None, None, None, None)
    mock_construct_weights.return_value = [("test", MagicMock(spec=UpdateGenerator))]
    mock_construct_gan_from_checkpoint.return_value = (MagicMock(spec=nn.Module), MagicMock(spec=nn.Module), None, None)

    # Mock the construct_optimizers return value with mock optimizers
    mock_construct_optimizers.return_value = (mock_step2_args.g_opt, mock_step2_args.d_opt)
    mock_train.return_value = ({}, None, None, mock_stats)
    mock_focd.return_value = MagicMock(spec=FOCD)

    original_fid = MagicMock(spec=FID)
    mock_step2_args.g_crit = MagicMock(spec=UpdateGeneratorGAN)
    step_1_train_state = {"best_epoch": 1}

    train_step2_gan(mock_step2_args, mock_config, original_fid, step_1_train_state)

    mock_construct_classifier.assert_called_once()
    mock_construct_weights.assert_called_once()
    mock_construct_optimizers.assert_called()
