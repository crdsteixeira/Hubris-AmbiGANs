"""Module for creating an ensemble of models."""

import torch
from torch import nn

from src.classifier.pretrained import ClassifierMLP, ClassifierResnet
from src.classifier.simple_cnn import Classifier
from src.enums import ClassifierType, EnsembleType, OutputMethod
from src.models import ClassifierParams, TrainClassifierArgs


class Ensemble(nn.Module):
    """Class for creating an ensemble of models and combining their outputs."""

    def __init__(self, params: ClassifierParams) -> None:
        """Initialize ensemble model based on the provided parameters."""
        super().__init__()
        self.params = params

        # List of ensemble models, either pretrained ones or CNN's.
        if self.params.ensemble_type == EnsembleType.pretrained:
            self.train_models = False
            self.models = nn.ModuleList(
                [
                    ClassifierResnet(params),
                    ClassifierMLP(params),
                ],
            )
        elif self.params.ensemble_type == EnsembleType.cnn:
            # Generate models' parameters
            self.train_models = True
            self.cnn_list = self.params.nf
            # Check if cnn_list is of a type that has a length
            if isinstance(self.cnn_list, (list | tuple)):
                self.cnn_count = len(self.cnn_list)
            else:
                raise ValueError("cnn_list must be a list or tuple when ensemble_type is cnn")
            # Generate models
            self.models = nn.ModuleList(
                [
                    Classifier(
                        ClassifierParams(
                            type=ClassifierType.cnn,
                            img_size=self.params.img_size,
                            nf=cnn,
                            ensemble_type=None,
                            output_method=None,
                            n_classes=self.params.n_classes,
                            device=self.params.device,
                        )
                    )
                    for cnn in self.cnn_list
                ]
            )

        # Output method for ensemble
        if self.params.output_method == OutputMethod.meta_learner:
            # Multi-probability combinator
            self.optimize = True
            self.predictor = nn.Sequential(
                nn.Linear(len(self.models), len(self.models)),
                nn.ReLU(),
                nn.Linear(len(self.models), len(self.models)),
                nn.ReLU(),
                nn.Linear(len(self.models), 1),
                nn.Sigmoid(),
            )
        elif self.params.output_method == OutputMethod.mean:
            # Mean combinator
            self.optimize = False
            self.predictor = nn.Sequential(
                nn.Flatten(),
                nn.AvgPool1d(len(self.models)),
            )
        elif self.params.output_method == OutputMethod.linear:
            # Linear combinator
            self.predictor = nn.Sequential(
                nn.Linear(len(self.models), 1),
                nn.Flatten(),
                nn.Sigmoid(),
            )
        elif self.params.output_method == OutputMethod.identity:
            self.optimize = False
            # Raw outputs with no combination
            self.predictor = nn.Sequential(nn.Identity())

    def forward(
        self, x: torch.Tensor, output_feature_maps: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Perform forward pass through the ensemble and optionally return feature maps."""
        # Initialize the output as a list to avoid shape mismatches in stacking
        outputs = []
        feature_maps = []

        # Pass input through each model in the ensemble
        for model in self.models:
            # Get the output of each model
            out = model(x.clone(), output_feature_maps=output_feature_maps)
            if output_feature_maps and isinstance(out, tuple):
                feature_maps.append(out[0])  # Assume first element in tuple is feature maps
                out = out[1]  # Assume second element is the final output
            outputs.append(out.unsqueeze(-1))  # Add a new dimension to align for stacking

        # Concatenate outputs along the last dimension
        output = torch.cat(outputs, dim=-1)

        # Combine the outputs using the predictor
        combined_output = self.predictor(output).squeeze(-1)

        if output_feature_maps:
            return combined_output, feature_maps

        return combined_output

    def train_helper(
        self,
        _: None,
        X: torch.Tensor,
        Y: torch.Tensor,
        crit: nn.Module,
        acc_fun: nn.Module,
        early_acc: float = 1.00,
        __: TrainClassifierArgs | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Help to train ensemble models as utility function."""
        chunks = list(
            zip(
                torch.tensor_split(X, len(self.models) + 1),
                torch.tensor_split(Y, len(self.models) + 1),
            )
        )[1:]
        loss_overall = 0
        acc = 0

        for i, chunk in enumerate(chunks):
            x, y = chunk[0], chunk[1]
            y_hat = self.models[i](x, output_feature_maps=False)
            loss = crit(y_hat, y)
            loss_overall += loss
            local_acc = acc_fun(y_hat, y, avg=False)
            acc += local_acc
            if early_acc >= (local_acc / len(y)):
                loss.backward()

        return loss_overall / len(self.models), acc / len(self.models)

    def optimize_helper(
        self,
        _: None,
        X: torch.Tensor,
        Y: torch.Tensor,
        crit: nn.Module,
        acc_fun: nn.Module,
        early_acc: float = 1.00,
        __: TrainClassifierArgs | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Help to optimize ensemble models as utility function."""
        chunks = list(
            zip(
                torch.tensor_split(X, len(self.models) + 1),
                torch.tensor_split(Y, len(self.models) + 1),
            )
        )[0]
        x, y = chunks[0], chunks[1]

        for p in self.models.parameters():
            p.requires_grad = False

        y_hat = self.forward(x)
        loss = crit(y_hat, y)
        acc = acc_fun(y_hat, y, avg=False)

        if (self.params.output_method != OutputMethod.mean) and (early_acc >= (acc / len(y))):
            loss.backward()

        return loss, acc
