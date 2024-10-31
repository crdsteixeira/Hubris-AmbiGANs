"""Module to test classifier."""

import pytest

from src.classifier.construct_classifier import construct_classifier
from src.classifier.ensemble import Ensemble
from src.classifier.my_mlp import Classifier as MyMLP
from src.classifier.simple_cnn import Classifier as SimpleCNN
from src.enums import EnsembleType, OutputMethod
from src.models import ClassifierParams  # Import the required enums


def test_construct_classifier_cnn() -> None:
    """Test constructing a CNN classifier."""
    params: ClassifierParams = ClassifierParams(type="cnn", img_size=(3, 64, 64), n_classes=10, nf=[32, 64])
    classifier = construct_classifier(params)
    assert isinstance(classifier, SimpleCNN)


def test_construct_classifier_mlp() -> None:
    """Test constructing an MLP classifier."""
    params: ClassifierParams = ClassifierParams(type="mlp", img_size=(3, 64, 64), n_classes=10, nf=32)
    classifier = construct_classifier(params)
    assert isinstance(classifier, MyMLP)


def test_construct_classifier_ensemble_mean() -> None:
    """Test constructing an ensemble classifier with mean output method."""
    params: ClassifierParams = ClassifierParams(
        type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=[[32, 16, 8], [32, 16, 8]]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, Ensemble)
    assert classifier.params.ensemble_type == EnsembleType.cnn
    assert classifier.params.output_method == OutputMethod.mean


def test_construct_classifier_ensemble_linear() -> None:
    """Test constructing an ensemble classifier with linear output method."""
    params: ClassifierParams = ClassifierParams(
        type="ensemble:cnn:linear", img_size=(3, 64, 64), n_classes=10, nf=[[32], [16]]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, Ensemble)
    assert classifier.params.ensemble_type == EnsembleType.cnn
    assert classifier.params.output_method == OutputMethod.linear


def test_construct_classifier_ensemble_meta_learner() -> None:
    """Test constructing an ensemble classifier with meta-learner output method."""
    params: ClassifierParams = ClassifierParams(
        type="ensemble:cnn:meta-learner", img_size=(3, 64, 64), n_classes=10, nf=[[32], [16]]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, Ensemble)
    assert classifier.params.ensemble_type == EnsembleType.cnn
    assert classifier.params.output_method == OutputMethod.meta_learner


def test_invalid_classifier_type() -> None:
    """Test constructing classifier with invalid type."""
    with pytest.raises(ValueError):
        params: ClassifierParams = ClassifierParams(type="unknown", img_size=(3, 64, 64), n_classes=10, nf=[[32]])
        construct_classifier(params)


def test_invalid_ensemble_format() -> None:
    """Test constructing ensemble classifier with invalid format."""
    with pytest.raises(ValueError):
        params: ClassifierParams = ClassifierParams(type="ensemble:cnn", img_size=(3, 64, 64), n_classes=10, nf=[[32]])
        construct_classifier(params)


def test_invalid_ensemble_type() -> None:
    """Test constructing ensemble classifier with invalid ensemble type."""
    with pytest.raises(ValueError):
        params: ClassifierParams = ClassifierParams(
            type="ensemble:invalid:mean", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
        )
        construct_classifier(params)


def test_invalid_output_method() -> None:
    """Test constructing ensemble classifier with invalid output method."""
    with pytest.raises(ValueError):
        params: ClassifierParams = ClassifierParams(
            type="ensemble:cnn:invalid", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
        )
        construct_classifier(params)


def test_invalid_nf_length_for_ensemble() -> None:
    """Test constructing ensemble classifier with invalid nf length."""
    with pytest.raises(ValueError):
        params: ClassifierParams = ClassifierParams(type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=32)
        assert isinstance(params, ClassifierParams)

    with pytest.raises(AssertionError):
        params: ClassifierParams = ClassifierParams(
            type="ensemble:cnn:mean",
            img_size=(3, 64, 64),
            n_classes=10,
            nf=[[32], [64]],
        )
        ensemble = construct_classifier(params)
        assert len(ensemble.models) != len(params.nf)


def test_invalid_nf_for_pretrained_ensemble() -> None:
    """Test constructing pretrained ensemble classifier with invalid nf."""
    with pytest.raises(ValueError):
        params: ClassifierParams = ClassifierParams(
            type="ensemble:pretrained:mean", img_size=(3, 64, 64), n_classes=10, nf=[32]
        )
        assert isinstance(params, ClassifierParams)


def test_cnn_nf_list_of_ints() -> None:
    """Test nf parameter is list of ints for CNN classifier."""
    params: ClassifierParams = ClassifierParams(type="cnn", img_size=(3, 64, 64), n_classes=10, nf=[32, 64])
    assert isinstance(params.nf, list)


def test_mlp_nf_single_int() -> None:
    """Test nf parameter is a single int for MLP classifier."""
    params: ClassifierParams = ClassifierParams(type="mlp", img_size=(3, 64, 64), n_classes=10, nf=32)
    assert isinstance(params.nf, int)


def test_ensemble_cnn_nf_list_of_lists() -> None:
    """Test nf parameter is a list of lists for ensemble CNN classifier."""
    params: ClassifierParams = ClassifierParams(
        type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=[[32], [64]]
    )
    assert isinstance(params.nf, list)


def test_ensemble_pretrained_nf_none() -> None:
    """Test nf parameter is None for pretrained ensemble classifier."""
    params: ClassifierParams = ClassifierParams(
        type="ensemble:pretrained:mean", img_size=(1, 64, 64), n_classes=10, nf=None
    )
    assert params.nf is None


def test_invalid_cnn_nf() -> None:
    """Test invalid nf parameter for CNN classifier."""
    with pytest.raises(ValueError):
        ClassifierParams(type="cnn", img_size=(3, 64, 64), n_classes=10, nf=32)


def test_invalid_mlp_nf() -> None:
    """Test invalid nf parameter for MLP classifier."""
    with pytest.raises(ValueError):
        ClassifierParams(type="mlp", img_size=(3, 64, 64), n_classes=10, nf=[32, 64])


def test_invalid_ensemble_cnn_nf() -> None:
    """Test invalid nf parameter for ensemble CNN classifier."""
    with pytest.raises(ValueError):
        ClassifierParams(type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=[32])
