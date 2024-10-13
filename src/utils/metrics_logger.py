"""Module for logging Metrics to wandb"""

import matplotlib.pyplot as plt
import wandb
from typing import Optional


class MetricsLogger:
    """Class for logging training metrics, including iteration and epoch metrics."""

    def __init__(self, prefix: Optional[str] = None, log_epoch: bool = True) -> None:
        """Initialize the MetricsLogger with a prefix and log epoch flag."""
        self.prefix = prefix
        self.iteration_metrics = []
        self.running_stats = {}
        self.it_counter = {}
        self.stats = {}
        self.log_epoch = log_epoch
        self.log_dict = {}
        self.epoch = 1

    def add_media_metric(self, name: str) -> None:
        """Define a media metric for WandB logging."""
        wandb.define_metric(name, step_metric=self.apply_prefix("epoch"))
        self.log_dict[self.apply_prefix(name)] = None

    def log_image(self, name: str, image, caption: Optional[str] = None) -> None:
        """Log an image to WandB with an optional caption."""
        self.log_dict[self.apply_prefix(name)] = wandb.Image(image, caption=caption)

    def log_plot(self, name: str) -> None:
        """Log a plot to WandB."""
        wandb.log({self.apply_prefix(name): plt})

    def apply_prefix(self, name: str) -> str:
        """Apply a prefix to a metric name, if a prefix is defined."""
        return f"{self.prefix}/{name}" if self.prefix is not None else name

    def add(self, name: str, iteration_metric: bool = False) -> None:
        """Add a metric to the logger, optionally marking it as an iteration metric."""
        self.stats[name] = []
        wandb.define_metric(self.apply_prefix(name), step_metric=self.apply_prefix("epoch"))
        self.log_dict[self.apply_prefix(name)] = None

        if iteration_metric:
            self.iteration_metrics.append(name)
            self.stats[f"{name}_per_it"] = []
            self.running_stats[name] = 0
            self.it_counter[name] = 0

    def reset_it_metrics(self) -> None:
        """Reset iteration metrics to zero."""
        for name in self.iteration_metrics:
            self.running_stats[name] = 0
            self.it_counter[name] = 0

    def update_it_metric(self, name: str, value: float) -> None:
        """Update the value of an iteration metric."""
        self.running_stats[name] += value
        self.it_counter[name] += 1

    def update_epoch_metric(self, name: str, value: float, prnt: bool = False) -> None:
        """Update the value of an epoch metric and optionally print it."""
        self.stats[name].append(value)
        self.log_dict[self.apply_prefix(name)] = value

        if prnt:
            print(name, " = ", value)

    def finalize_epoch(self) -> None:
        """Finalize epoch metrics by computing averages and logging them."""
        # Compute average of iteration metrics per epoch
        for name in self.iteration_metrics:
            epoch_value = self.running_stats[name] / self.it_counter[name]
            self.stats[name].append(epoch_value)
            self.log_dict[self.apply_prefix(name)] = epoch_value

            if self.log_epoch:
                print(name, " = ", epoch_value)

        self.log_dict[self.apply_prefix("epoch")] = self.epoch
        self.epoch += 1

        wandb.log(self.log_dict, commit=True)