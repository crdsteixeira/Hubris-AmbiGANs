#!/usr/bin/env bash

# Run the classifier script
poetry run python -m src.classifier.classifier_cli \
    --data_dir="$FILESDIR/data" \
    --out_dir="$FILESDIR/models" \
    --name="test_classifier_cicd" \
    --batch_size=64 \
    --c_type="cnn" \
    --epochs=1 \
    --lr="0.01" \
    --nf=1 \
    --seed=42 \
    --device="cpu" \
    --dataset_name="mnist" \
    --pos_class=7 \
    --neg_class=1
