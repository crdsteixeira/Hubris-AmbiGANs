from src.classifier.simple_cnn import Classifier as SimpleCNN
from src.classifier.my_mlp import Classifier as MyMLP
from src.classifier.ensemble import Ensemble
from src.classifier.classifier_cache import ClassifierCache
from src.models import ClassifierParams, ClassifierType, EnsembleType


def construct_classifier(params: ClassifierParams, device=None):
    if params.type == ClassifierType.cnn:
        C = SimpleCNN(params)
    elif params.type == ClassifierType.mlp:
        C = MyMLP(params)
    elif params.type == ClassifierType.ensemble:
        # Directly pass the entire params object to the Ensemble class
        C = Ensemble(params)
        
        # Perform additional validation if needed
        if params.ensemble_type == EnsembleType.cnn and len(params.nf) != len(C.models):
            raise ValueError(f"nf length {len(params.nf)} does not match the number of ensemble models {len(C.models)}")
    else:
        raise ValueError(f"Unknown classifier type: {params.type}")

    return C.to(device)
