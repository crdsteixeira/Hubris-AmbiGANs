#!/usr/bin/env bash

# Resolve project root from script location
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Change to project root directory
cd "$ROOT_DIR" || exit 1

# Run the classifier script
python -m src.classifier.classifier_cli \
    --data_dir="$ROOT_DIR/data" \
    --out_dir="$ROOT_DIR/out" \
    --name="test_classifier_cicd" \
    --batch_size=64 \
    --c_type="ensemble" \
    --epochs=1 \
    --lr="0.01" \
    --nf=2 \
    --seed=42 \
    --device="cpu" \
    --dataset_name="mnist" \
    --pos_class=7 \
    --neg_class=1 \
    --ensemble_type="cnn" \
    --ensemble_output_method="mean"
