"""Module to construct classifier."""

from torch import nn

from src.classifier.ensemble import Ensemble
from src.classifier.my_mlp import Classifier as MyMLP
from src.classifier.simple_cnn import Classifier as SimpleCNN
from src.models import ClassifierType, EnsembleType, TrainClassifierArgs


def construct_classifier(params: TrainClassifierArgs) -> nn.Module:
    """Construct a classifier model based on the given parameters."""
    if params.type == ClassifierType.cnn:
        C = SimpleCNN(params)
    elif params.type == ClassifierType.mlp:
        C = MyMLP(params)
    elif params.type == ClassifierType.ensemble:
        # Directly pass the entire params object to the Ensemble class
        C = Ensemble(params)

        # Perform additional validation if needed
        if (
            params.ensemble_type == EnsembleType.cnn
            and isinstance(params.nf, list)  # Ensure `params.nf` is a list before calling len()
            and len(params.nf) != len(C.models)
        ):
            raise ValueError(f"nf length {len(params.nf)} does not match the number of ensemble models {len(C.models)}")
        
        if len(C.models) <= 1:  # Ensure ensemble has more than one model
            raise ValueError(f"Ensemble must have more than one model, but got {len(C.models)}")
    else:
        raise ValueError(f"Unknown classifier type: {params.type}")

    return C.to(params.device.value)
