"""Module to test checkpoint."""

import json
import os
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from src.enums import (
    ArchitectureType,
    ClassifierType,
    DatasetNames,
    DeviceType,
    LossType,
)
from src.gan.loss import DiscriminatorLoss
from src.gan.update_g import UpdateGenerator
from src.models import (
    CLTrainArgs,
    ConfigArchitecture,
    ConfigBinary,
    ConfigDatasetParams,
    ConfigGAN,
    ConfigLoss,
    ConfigModel,
    ConfigOptimizer,
    ConfigStep1,
    ConfigStep2,
    ConfigTrain,
    FIDMetricsParams,
    GANTrainArgs,
    TrainClassifierArgs,
    TrainingStats,
)
from src.utils.checkpoint import (
    checkpoint,
    checkpoint_gan,
    checkpoint_image,
    construct_classifier_from_checkpoint,
    construct_gan_from_checkpoint,
    load_checkpoint,
)


@pytest.fixture
def mock_model() -> nn.Module:
    """Fixture to provide a mock model."""
    return MagicMock(spec=nn.Module)


@pytest.fixture
def mock_optimizer() -> optim.Optimizer:
    """Fixture to provide a mock optimizer."""
    return MagicMock(spec=optim.Optimizer)


@pytest.fixture
def mock_train_stats() -> TrainingStats:
    """Fixture to provide mock training statistics."""
    return TrainingStats(loss=0.5, accuracy=0.8, epoch=10)


@pytest.fixture
def mock_cl_train_args() -> CLTrainArgs:
    """Fixture to provide mock classifier training arguments."""
    return CLTrainArgs(
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        device="cpu",
        out_dir="mock_dir",
        dataset_name="mnist",
        pos_class=1,
        neg_class=7,
    )


@pytest.fixture
def mock_train_classifier_args() -> TrainClassifierArgs:
    """Fixture to provide mock training classifier arguments."""
    return TrainClassifierArgs(
        type=ClassifierType.cnn,
        img_size=(3, 64, 64),
        n_classes=10,
        nf=[1],
        ensemble_type=None,
        device=DeviceType.cpu,
        epochs=20,
        early_acc=1.0,
        out_dir="mock_dir",
        batch_size=64,
        lr=0.0005,
        seed=None,
        early_stop=None,
    )


@patch("torch.save")
def test_checkpoint(mock_save, mock_model, mock_train_classifier_args, mock_train_stats, mock_cl_train_args) -> None:
    """Test saving a model checkpoint."""
    output_dir = tempfile.mkdtemp()
    result_dir = checkpoint(
        model=mock_model,
        model_name="mock_model",
        model_params=mock_train_classifier_args,
        train_stats=mock_train_stats,
        args=mock_cl_train_args,
        output_dir=output_dir,
    )

    # Check if the directory was created
    assert os.path.exists(os.path.join(output_dir, "mock_model"))

    mock_save.assert_called_once()
    assert result_dir == os.path.join(output_dir, "mock_model")


@patch("torch.load")
def test_load_checkpoint(mock_load, mock_model, mock_optimizer) -> None:
    """Test loading a model checkpoint from disk."""
    mock_load.return_value = {
        "state": mock_model.state_dict(),
        "optimizer": mock_optimizer.state_dict(),
    }
    load_checkpoint("mock_path", mock_model, optimizer=mock_optimizer)

    mock_model.load_state_dict.assert_called_once()
    mock_optimizer.load_state_dict.assert_called_once()


@patch("torch.load")
@patch("src.utils.checkpoint.construct_classifier")
def test_construct_classifier_from_checkpoint(
    mock_construct_classifier, mock_load, mock_train_classifier_args, mock_cl_train_args
) -> None:
    """Test constructing a classifier model from a saved checkpoint."""
    mock_load.return_value = {
        "name": "mock_classifier",
        "params": mock_train_classifier_args,
        "state": {},
        "stats": {},
        "args": mock_cl_train_args,
        "optimizer": MagicMock(spec=optim.Optimizer).state_dict(),
    }

    model, model_params, stats, args, optimizer = construct_classifier_from_checkpoint("mock_path")

    mock_construct_classifier.assert_called_once()
    assert model == mock_construct_classifier.return_value
    assert isinstance(model_params, TrainClassifierArgs)
    assert isinstance(args, CLTrainArgs)
    assert optimizer is None or isinstance(optimizer, optim.Optimizer)


@patch("torch.load")
@patch("src.utils.checkpoint.construct_gan")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="""{
       "config": {
            "project": "project_name",
            "name": "gan_config",
            "out_dir": "output_dir",
            "data_dir": "data_dir",
            "fid_stats_path": "fid_path",
            "fixed_noise": "fixed_noise_str",
            "test_noise": "test_noise_str", 
            "dataset": {
                "name": "mnist", 
                "binary": {"pos": 1, "neg": 7}  
            },
            "model": {
                "z_dim": 128,
                "architecture": {
                    "name": "dcgan",
                    "g_filter_dim": 64,
                    "d_filter_dim": 64,
                    "g_num_blocks": 3,
                    "d_num_blocks": 2
                },
                "loss": {
                    "name": "ns"
                }
            },
            "optimizer": {
                "lr": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999
            },
            "train": {
                "step_1": {
                    "epochs": 10,
                    "batch_size": 32
                },
                "step_2": {
                    "epochs": 10,
                    "batch_size": 32,
                    "classifier": ["classifier_path"],
                    "weight": []
                }
            }
        },
        "gen_params": {"image_size": [3, 64, 64]},  
        "dis_params": {"image_size": [3, 64, 64]}
}""",
)
def test_construct_gan_from_checkpoint(mock_open_file, mock_construct_gan, mock_load) -> None:
    """Test constructing a GAN model from a saved checkpoint."""
    mock_load.side_effect = [
        {
            "state": {},
            "optimizer": {
                "state": {0: {}},  # Mocked state (just an empty dict for simplicity)
                "param_groups": [{"lr": 0.0002, "betas": (0.5, 0.999), "params": [0]}],  # Added "params" key
            },
        },  # Generator checkpoint
        {
            "state": {},
            "optimizer": {
                "state": {0: {}},  # Mocked state (just an empty dict for simplicity)
                "param_groups": [{"lr": 0.0002, "betas": (0.5, 0.999), "params": [0]}],  # Added "params" key
            },
        },  # Discriminator checkpoint
    ]

    # Create a mock generator and discriminator with parameters
    mock_G = MagicMock(spec=nn.Module)
    mock_D = MagicMock(spec=nn.Module)

    # Mock the parameters method to return a non-empty list of parameters
    mock_param = MagicMock(spec=torch.Tensor)
    mock_G.parameters.return_value = [mock_param]
    mock_D.parameters.return_value = [mock_param]

    # Mock the GAN construction method to return the mocked generator and discriminator
    mock_construct_gan.return_value = (mock_G, mock_D)

    G, D, g_optim, d_optim = construct_gan_from_checkpoint("mock_path")

    mock_open_file.assert_called_with(os.path.join("mock_path", "config.json"), encoding="utf-8")
    assert isinstance(G, nn.Module)
    assert isinstance(G, nn.Module)
    assert isinstance(D, nn.Module)
    assert isinstance(g_optim, optim.Optimizer)
    assert isinstance(d_optim, optim.Optimizer)


@patch("torch.save")
@patch("json.dump")
def test_checkpoint_gan(mock_json_dump, mock_save):
    """Test saving a GAN checkpoint."""
    output_dir = tempfile.mkdtemp()

    # Mock Generator and Discriminator as nn.Module instances
    mock_G = MagicMock(spec=nn.Module)
    mock_D = MagicMock(spec=nn.Module)

    # Mock Optimizers as optim.Optimizer instances
    mock_g_opt = MagicMock(spec=optim.Optimizer)
    mock_d_opt = MagicMock(spec=optim.Optimizer)

    # Create instances for other required arguments
    mock_g_updater = MagicMock(spec=UpdateGenerator)
    mock_d_crit = MagicMock(spec=DiscriminatorLoss)
    mock_test_noise = torch.randn(128, 100)  # Example tensor for test noise
    mock_fid_metrics = MagicMock(spec=FIDMetricsParams)
    mock_dataset = MagicMock(spec=Dataset)

    # Create a ConfigGAN instance with all required fields
    mock_config = ConfigGAN(
        project="mock_project",
        name="mock_name",
        out_dir="mock_out_dir",
        data_dir="mock_data_dir",
        fid_stats_path="mock_fid_path",
        fixed_noise="mock_fixed_noise",
        test_noise="mock_test_noise",
        compute_fid=True,
        device=DeviceType.cpu,
        num_workers=4,
        num_runs=1,
        step_1_seeds=[42],
        step_2_seeds=None,
        dataset=ConfigDatasetParams(
            name=DatasetNames.mnist,
            binary=ConfigBinary(pos=1, neg=7),
        ),
        model=ConfigModel(
            z_dim=128,
            architecture=ConfigArchitecture(
                name=ArchitectureType.dcgan, g_filter_dim=64, d_filter_dim=64, g_num_blocks=3, d_num_blocks=2
            ),
            loss=ConfigLoss(name=LossType.ns),
        ),
        optimizer=ConfigOptimizer(lr=0.0002, beta1=0.5, beta2=0.999),
        train=ConfigTrain(
            step_1=ConfigStep1(
                epochs=10,
                batch_size=32,
                checkpoint_every=5,
                disc_iters=1,
            ),
            step_2=ConfigStep2(epochs=10, batch_size=32, classifier=["classifier_path"], weight=[]),
        ),
    )

    # Create instances of GANTrainArgs with the proper mock objects
    mock_params = GANTrainArgs(
        G=mock_G,
        g_opt=mock_g_opt,
        g_updater=mock_g_updater,
        D=mock_D,
        d_opt=mock_d_opt,
        d_crit=mock_d_crit,
        test_noise=mock_test_noise,
        fid_metrics=mock_fid_metrics,
        n_disc_iters=1,
        epochs=20,
        early_acc=1.0,
        out_dir=output_dir,
        batch_size=64,
        lr=0.0005,
        seed=None,
        device=DeviceType.cpu,
        dataset=mock_dataset,
    )

    # Mock state and stats
    mock_state = {}
    mock_stats = {}

    # Use the proper ConfigGAN instance
    result_dir = checkpoint_gan(
        params=mock_params,
        state=mock_state,
        stats=mock_stats,
        config=mock_config,
        output_dir=output_dir,
        epoch=0,
    )

    # Assertions
    mock_save.assert_called()
    mock_json_dump.assert_called()
    assert result_dir == os.path.join(output_dir, "00")
