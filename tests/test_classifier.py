import pytest

from src.classifier import construct_classifier
from src.classifier.ensemble import Ensemble
from src.classifier.my_mlp import Classifier as MyMLP
from src.classifier.simple_cnn import Classifier as SimpleCNN
from src.models import ClassifierParams  # Import the required enums
from src.models import EnsembleType, OutputMethod


def test_construct_classifier_cnn():
    params = ClassifierParams(
        type="cnn", img_size=(3, 64, 64), n_classes=10, nf=[32, 64]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, SimpleCNN)


def test_construct_classifier_mlp():
    params = ClassifierParams(
        type="mlp", img_size=(3, 64, 64), n_classes=10, nf=32)
    classifier = construct_classifier(params)
    assert isinstance(classifier, MyMLP)


def test_construct_classifier_ensemble_mean():
    params = ClassifierParams(
        type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=[[32, 16, 8]]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, Ensemble)
    assert classifier.ensemble_type == EnsembleType.cnn  # Verify ensemble type
    assert classifier.output_method == OutputMethod.mean  # Verify output method


def test_construct_classifier_ensemble_linear():
    params = ClassifierParams(
        type="ensemble:cnn:linear", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, Ensemble)
    assert classifier.ensemble_type == EnsembleType.cnn  # Verify ensemble type
    assert classifier.output_method == OutputMethod.linear  # Verify output method


def test_construct_classifier_ensemble_meta_learner():
    params = ClassifierParams(
        type="ensemble:cnn:meta-learner", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
    )
    classifier = construct_classifier(params)
    assert isinstance(classifier, Ensemble)
    assert classifier.ensemble_type == EnsembleType.cnn  # Verify ensemble type
    assert classifier.output_method == OutputMethod.meta_learner  # Verify output method


def test_invalid_classifier_type():
    with pytest.raises(ValueError):
        params = ClassifierParams(
            type="unknown", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
        )
        construct_classifier(params)


def test_invalid_ensemble_format():
    with pytest.raises(ValueError):
        params = ClassifierParams(
            type="ensemble:cnn", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
        )
        construct_classifier(params)  # Missing output method


def test_invalid_ensemble_type():
    with pytest.raises(ValueError):
        params = ClassifierParams(
            type="ensemble:invalid:mean", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
        )
        construct_classifier(params)  # Invalid ensemble type


def test_invalid_output_method():
    with pytest.raises(ValueError):
        params = ClassifierParams(
            type="ensemble:cnn:invalid", img_size=(3, 64, 64), n_classes=10, nf=[[32]]
        )
        construct_classifier(params)  # Invalid output method


def test_invalid_nf_length_for_ensemble():
    with pytest.raises(ValueError):
        params = ClassifierParams(
            type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=32
        )  # Invalid nf type
        assert isinstance(params, ClassifierParams)

    with pytest.raises(AssertionError):
        params = ClassifierParams(
            type="ensemble:cnn:mean",
            img_size=(3, 64, 64),
            n_classes=10,
            nf=[[32], [64]],
        )
        ensemble = construct_classifier(params)
        assert len(ensemble.models) != len(params.nf)


def test_invalid_nf_for_pretrained_ensemble():
    with pytest.raises(ValueError):
        params = ClassifierParams(
            type="ensemble:pretrained:mean", img_size=(3, 64, 64), n_classes=10, nf=[32]
        )  # Invalid nf
        assert isinstance(
            params, ClassifierParams
        )  # Ensure it raises an error before instantiation


def test_cnn_nf_list_of_ints():
    params = ClassifierParams(
        type="cnn", img_size=(3, 64, 64), n_classes=10, nf=[32, 64]
    )
    assert isinstance(params.nf, list)


def test_mlp_nf_single_int():
    params = ClassifierParams(
        type="mlp", img_size=(3, 64, 64), n_classes=10, nf=32)
    assert isinstance(params.nf, int)


def test_ensemble_cnn_nf_list_of_lists():
    params = ClassifierParams(
        type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=[[32], [64]]
    )
    assert isinstance(params.nf, list)


def test_ensemble_pretrained_nf_none():
    params = ClassifierParams(
        type="ensemble:pretrained:mean", img_size=(1, 64, 64), n_classes=10, nf=None
    )
    assert params.nf is None


def test_invalid_cnn_nf():
    with pytest.raises(ValueError):
        ClassifierParams(
            type="cnn", img_size=(3, 64, 64), n_classes=10, nf=32
        )  # nf must be list of ints


def test_invalid_mlp_nf():
    with pytest.raises(ValueError):
        ClassifierParams(
            type="mlp", img_size=(3, 64, 64), n_classes=10, nf=[32, 64]
        )  # nf must be single int


def test_invalid_ensemble_cnn_nf():
    with pytest.raises(ValueError):
        ClassifierParams(
            type="ensemble:cnn:mean", img_size=(3, 64, 64), n_classes=10, nf=[32]
        )  # nf must be list of lists
