#!/usr/bin/env bash

# Run the classifier script
poetry run python -m src.classifier.classifier_cli \
    --data_dir="$FILESDIR/data" \
    --out_dir="$FILESDIR/out" \
    --name="test_classifier_cicd" \
    --batch_size=32 \
    --c_type="cnn" \
    --epochs=1 \
    --nf=1 \
    --seed=42 \
    --device="cpu" \
    --dataset_name="mnist" \
    --pos_class=1 \
    --neg_class=7
