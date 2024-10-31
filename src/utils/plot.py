"""Module with utility functions for plotting."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_loss(data: dict, out_path: str) -> None:
    """Plot the generator and discriminator losses."""
    plt.plot(data["G_losses_epoch"], label="G loss")
    plt.plot(data["D_losses_epoch"], label="D loss")

    if "term_1_epoch" in data:
        plt.plot(np.array(data["term_1_epoch"]), label="term_1")

    if "term_2_epoch" in data:
        plt.plot(np.array(data["term_2_epoch"]), label="term_2")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.legend()
    plt.savefig(os.path.join(out_path, "loss.png"))
    plt.clf()


def plot_d_outputs(data: dict, out_path: str) -> None:
    """Plot discriminator outputs during training."""
    plt.plot(data["D_x_epoch"], label="D(x)")
    plt.plot(data["D_G_z1_epoch"], label="D(G(z)) 1")
    plt.plot(data["D_G_z2_epoch"], label="D(G(z)) 2")
    plt.xlabel("epoch")
    plt.ylabel("output")
    plt.title("d outputs")
    plt.legend()
    plt.savefig(os.path.join(out_path, "d_outputs.png"))
    plt.clf()


def plot_d_accuracy(data: dict, out_path: str) -> None:
    """Plot discriminator accuracy for real and fake samples."""
    plt.plot(data["D_acc_real_epoch"], label="D acc real")
    plt.plot(data["D_acc_fake_1_epoch"], label="D acc fake 1")
    plt.plot(data["D_acc_fake_2_epoch"], label="D acc fake 2")
    plt.xlabel("epoch")
    plt.ylabel("output")
    plt.title("d accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_path, "d_accuracy.png"))
    plt.clf()


def plot_fid(data: dict, out_path: str) -> None:
    """Plot FID score."""
    plt.plot(data["fid"], label="FID")
    plt.xlabel("epoch")
    plt.ylabel("fid")
    plt.title("fid")
    plt.legend()
    plt.savefig(os.path.join(out_path, "fid.png"))
    plt.clf()


def plot_additional_metrics(data: dict, out_path: str) -> None:
    """Plot additional metrics if available (FOCD, confusion distance)."""
    if "focd" in data:
        plt.plot(data["focd"], label="F*D")
        plt.xlabel("epoch")
        plt.ylabel("f*d")
        plt.title("f*d")
        plt.legend()
        plt.savefig(os.path.join(out_path, "f*d.png"))
        plt.clf()

    if "conf_dist" in data:
        plt.plot(data["conf_dist"], label="conf_dist")
        plt.xlabel("epoch")
        plt.ylabel("conf_dist")
        plt.title("conf_dist")
        plt.legend()
        plt.savefig(os.path.join(out_path, "conf_dist.png"))
        plt.clf()


def plot_train_summary(data: dict, out_path: str) -> None:
    """Plot training summary including losses and metrics."""
    os.makedirs(out_path, exist_ok=True)

    plot_loss(data, out_path)
    plot_d_outputs(data, out_path)
    plot_d_accuracy(data, out_path)
    plot_fid(data, out_path)
    plot_additional_metrics(data, out_path)


def plot_metrics(data: pd.DataFrame, path: str, C_name: str) -> None:
    """Plot metrics and save to CSV and SVG files."""
    fid = data["fid"].to_numpy()
    cd = data["conf_dist"].to_numpy()

    costs = np.array(list(zip(fid, cd)))
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True

    size = [1.5 if pe else 1 for pe in is_efficient]
    data["pareto_efficient"] = is_efficient

    data.to_csv(os.path.join(path, f"metrics_{C_name}.csv"))

    sns.scatterplot(
        data=data,
        x="fid",
        y="conf_dist",
        hue="weight",
        style="s1_epochs",
        palette="deep",
        size=size,
    )
    plt.savefig(os.path.join(path, f"metrics_{C_name}.svg"))
    plt.close()
