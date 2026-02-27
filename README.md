# AmbiGANs

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
![CI](https://github.com/crdsteixeira/Paper-Hubris-AmbiGANs/actions/workflows/main.yml/badge.svg)


**Hubris benchmarking**: a methodology to evaluate overconfidence in machine learning models using ambiguous generative adversarial networks (AmbiGANs). This tool synthesizes realistic yet ambiguous images to measure model overconfidence.

We also propose the hubris metric to quantitatively measure the extent of model overconfidence when faced with ambiguous images. Testing on state-of-the-art pre-trained models (ConvNext, ViT, EfficientNet and Swin Transformer) using binarized versions of MNIST, Fashion-MNIST, and Pneumonia Chest X-ray datasets.

## Pre-conditions

* You must [install Poetry](https://python-poetry.org/docs/) for dependency management.
* You must have a [Weights & Biases account](https://wandb.ai/site) for logging training metrics.
* You must run the project in a CUDA supported OS.

## Setup environment

### Create the environment for the project

```ssh
poetry install
poetry shell
```

### Create your .env file

```bash
CUDA_VISIBLE_DEVICES=0
FILESDIR=<artifacts-file-directory>
ENTITY=<wandb-account-name-to-track-experiments>
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Project Structure

```
src/               # Main source code (gen_test_noise, classifier, gan, metrics, evaluation)
experiments/       # Configuration files (.yml) for experiments
data/              # Input datasets (MNIST, Fashion-MNIST, Chest X-ray)
scripts/           # Utility scripts
tests/             # Test files
```

## Run

## Dataset Information

- **Supported**: MNIST, Fashion-MNIST, Pneumonia Chest X-ray (binary classification)
- **Format**: Images organized by class in `data/<dataset_name>/` directory
- **Preparation**: Use official dataset sources; custom datasets must follow same structure

### 1. Experiment definition

Create a yml file containing the configurations for your experiment in `experiments/` folder. <br>
Example: 

```yml
# Global definitions
project: AmbiGAN
name: chest-xray
out_dir: out
data_dir: data
fixed_noise: 128
test_noise_seed: 15
device: cuda

# Number of runs to perform using same parameters but different seeds.
num_runs: 3

# Configuration
gen_test_noise: true
gen_pairwise_inception: true
gen_classifiers: true
gen_gan: true
gen_dataset: true
run_hubris_evaluation: true

# Dataset parameters, including positive and negative class.
dataset:
  name: chest-xray
  binary:
    pos: 1
    neg: 0

# Classifier list to be generated.
classifiers:
  - name: ensemble_mean_chestxray
    c_type: ensemble
    batch_size: 200
    epochs: 50
    lr: 0.001
    nf: 50
    seed: 32
    ensemble_type: cnn
    ensemble_output_method: mean

# Seed parameters for each run (step1 and step2).
step_1_seeds:
  - 42
  - 43
  - 44
step_2_seeds:
  - 34
  - 35
  - 36

# GAN model parameters
model:
  z_dim: 256
  architecture:
    name: dcgan
    g_filter_dim: 32
    d_filter_dim: 16
    g_num_blocks: 5
    d_num_blocks: 5
  loss:
    name: ns

# GAN optimizer parameters
optimizer:
  lr: 0.0002
  beta1: 0.00
  beta2: 0.999

# GAN training parameters
train:
  step_1:
    epochs: 100
    checkpoint_every: 100
    disc_iters: 1
    batch_size: 32
  step_2:
    epochs: 50
    checkpoint_every: 50
    disc_iters: 1
    batch_size: 32
    step_1_epochs:
      - last
    classifier:
      - ensemble_mean_chestxray
    weight:
      - gaussian:
          - alpha: 0.25
            var: 0.10
          - alpha: 0.50
            var: 0.10
          - alpha: 0.75
            var: 0.25
      - cd:
          alpha:
            - 10.0
            - 25.0
            - 50.0
```





### 2. Trigger AmbiGAN experiment

Run AmbiGAN to create images in the boundary between positive and negative classes.

```bash
poetry run python -m src --config experiments/ambigan_upscaling/chest-xray-initial.yml
```


### 3. Optional: Manually triggering individual process steps

| Step | Description | Command |
|------|-------------|---------|
| 1    | Create FID score for all pairs of numbers | `poetry run python -m src/gen_pairwise_inception --dataset chest-xray --device cuda` |
| 1.1  | Run for one pair only (e.g. 1v0) | `poetry run python -m src.metrics.fid.fid_cli --dataset chest-xray --pos 1 --neg 0` |
| 2.a    | Create set of 5 random classifiers with mean output estimator, given a pair of classes (e.g. 1v0) | `poetry run python -m src.classifier.classifier_cli --name="test_classifier" --batch_size=64 --c_type="ensemble" --epochs=20 --lr="0.01"  --nf=5 --seed=42  --device="cuda" --dataset_name="chest-xray" --pos_class=1 --neg_class=0 --ensemble_type="cnn"  --ensemble_output_method="mean"` |
| 2.b  | Create with linear output estimator | `poetry run python -m src.classifier.classifier_cli --name="test_classifier" --batch_size=64 --c_type="ensemble" --epochs=20 --lr="0.01"  --nf=5 --seed=42  --device="cuda" --dataset_name="chest-xray" --pos_class=1 --neg_class=0 --ensemble_type="cnn"  --ensemble_output_method="linear"`|
| 2.c  | Create with meta-learner output estimator | `poetry run python -m src.classifier.classifier_cli --name="test_classifier" --batch_size=64 --c_type="ensemble" --epochs=20 --lr="0.01"  --nf=5 --seed=42  --device="cuda" --dataset_name="chest-xray" --pos_class=1 --neg_class=0 --ensemble_type="cnn"  --ensemble_output_method="meta-learner"` |
| 2.d | Create with identity output estimator (no output combination) |  `poetry run python -m src.classifier.classifier_cli --name="test_classifier" --batch_size=64 --c_type="ensemble" --epochs=20 --lr="0.01"  --nf=5 --seed=42  --device="cuda" --dataset_name="chest-xray" --pos_class=1 --neg_class=0 --ensemble_type="cnn"  --ensemble_output_method="identity"` | 
| 3    | Create test noise | `poetry run python -m src.gen_test_noise --seed=42 --nz=128 --z-dim=256` |
| 4 | Train AmbiGAN | ` poetry run python -m src.gan.gan_cli --config="experiments/config_gan_test.yml"` | 

* In this example, in step 2 5 random CNNs will be created and trained during 20 epochs. Reproducibility of the CNN parameters is guaranteed by using the same random seed (--seed arg).
* For step 4, you need to create your experiments yml file. See example in `scripts/cicd/config_gan_test.yml`.

## Outputs

- **Generated images**: Saved in `out/<experiment_name>/` directory
- **Models**: GAN checkpoints and classifiers stored in experiment output folder
- **Metrics**: FID scores, hubris values, and evaluation results in experiment logs
- **W&B Logs**: All metrics tracked in Weights & Biases (if configured)

## Credits


This work started with the framework from previous developments of GASTeN from [luispcunha](https://github.com/luispcunha), published as [GASTeN: Generative Adversarial Stress Test Networks](https://link.springer.com/epdf/10.1007/978-3-031-30047-9_8?sharing_token=XGbq9zmVBDFAEaM4r1AAp_e4RwlQNchNByi7wbcMAY55SAL6inraGCkI72KOuzssTzewKWv51v_1pft7j7WJRbiAzL0vaTmG2vf4gs1QhnZ3lV72H7zSKLWQESXZjq5-1pg77WEnt2EHZaN2b51chvHsO6TW3tiGXSVhUgy87Ts%3D)

## All vs all experiments

Generate and run experiments for all combinations of binary classes (available datasets: MNIST, Fashion-MNIST):

```bash
python generate_configs.py --dataset mnist
./run_all_experiments.sh
```

## Ambiguity Evaluation

Evaluate model ambiguity multiclass classifier on companion datasets:

```bash
python -m src.evaluation.evaluation_ambiguity_cli experiments/ambiguity-evaluation/mnist.yaml
```

# Citation

```bibtex
@inproceedings{10.1007/978-3-032-05461-6_31,
author = {Teixeira, Cátia and Gomes, Inês and Soares, Carlos and van Rijn, Jan N.},
title = {Hubris Benchmarking with AmbiGANs: Assessing Model Overconfidence with&nbsp;Synthetic Ambiguous Data},
year = {2025},
doi = {10.1007/978-3-032-05461-6_31},
abstract = {The growing deployment of artificial intelligence in critical domains exposes a pressing challenge: how reliably models make predictions for ambiguous data without exhibiting overconfidence. We introduce hubris benchmarking, a methodology to evaluate overconfidence in machine learning models. The benchmark is based on a novel architecture, ambiguous generative adversarial networks (AmbiGANs), which are trained to synthesize realistic yet ambiguous datasets. We also propose the hubris metric to quantitatively measure the extent of model overconfidence when faced with these ambiguous images. We illustrate the usage of the methodology by estimating the hubris of state-of-the-art pre-trained models (ConvNext and ViT) on binarized versions of public datasets, including MNIST, Fashion-MNIST, and Pneumonia Chest X-ray. We found that, while ConvNext is on average 3\% more accurate than ViT, it often makes excessively confident predictions, on average by 10\% points higher than ViT. These results illustrate the usefulness of hubris benchmarking in high-stakes decision processes.},
booktitle = {Discovery Science: 28th International Conference, DS 2025, Ljubljana, Slovenia, September 23–25, 2025, Proceedings},
pages = {476–491},
keywords = {Synthetic Data Generation, Overconfidence, Generative Adversarial Networks, Responsible Artificial Intelligence, Computer Vision}
}
```
