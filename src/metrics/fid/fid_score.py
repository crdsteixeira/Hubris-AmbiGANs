"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code adapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import numpy as np
import torch
from scipy import linalg

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def load_statistics_from_path(path):
    """
    Load statistics from  npz file

    Params:
    -- path  : Path to .npz file

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    with np.load(path) as f:
        m, s = f["mu"][:], f["sigma"][:]

    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activations(images, feature_map_fn, batch_size=64, dims=2048, device="cpu"):
    """Calculates the activations of layer returned by feature_map_fn.

    Params:
    -- images           : Tensor of images (N images, C, H, W)
    -- feature_map_fn   : Function used to obtain layer
    -- batch_size       : Batch size of images for the model to process at once.
    -- dims             : Dimensionality of features returned by Inception (or other Classifier used)
    -- device           : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if batch_size > len(images):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(images)

    pred_arr = np.empty((len(images), dims))

    start_idx = 0
    num_batches = math.ceil(len(images) / batch_size)

    for _ in tqdm(range(num_batches)):
        # batch dim = (batch size, 3, 4, 4)
        batch = images[
            start_idx: start_idx + min(batch_size, len(images) - start_idx)
        ].to(device)

        with torch.no_grad():
            pred = feature_map_fn(batch, start_idx, batch.shape[0])

        pred = pred.cpu().numpy()

        pred_arr[start_idx: start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(
    images, feature_map_fn, batch_size=64, dims=2048, device="cpu"
):
    """Calculation of the statistics used by the FID.

    Params:
    -- images           : Tensor of images (N images, C, H, W)
    -- feature_map_fn   : Function used to obtain layer
    -- batch_size       : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims             : Dimensionality of features returned by feature_map_fn
    -- device           : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, feature_map_fn, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations_dataloader(dataloader, feature_map_fn, dims=2048, device="cpu"):
    """Calculates the activations of layer returned by feature_map_fn using images from a DataLoader.

    Params:
    -- dataloader       : PyTorch Dataloader
    -- feature_map_fn   : Function used to obtain layer
    -- dims             : Dimensionality of features returned by feature_map_fn
    -- device           : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    pred_arr = np.empty((len(dataloader.dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch[0].to(device)

        with torch.no_grad():
            pred = feature_map_fn(batch, start_idx, batch.shape[0])

        pred = pred.cpu().numpy()

        pred_arr[start_idx: start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics_dataloader(
    dataloader, feature_map_fn, dims=2048, device="cpu"
):
    """Calculation of the statistics used by the FID.

    Params:
    -- dataloader       : PyTorch Dataloader
    -- feature_map_fn   : Function used to obtain layer
    -- dims             : Dimensionality of features returned by feature_map_fn
    -- device           : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_dataloader(dataloader, feature_map_fn, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
