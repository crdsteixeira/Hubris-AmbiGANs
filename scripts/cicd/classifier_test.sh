#!/usr/bin/env bash

# Run the classifier script
python -m src.classifier.classifier_cli \
    --data_dir="$FILESDIR/data" \
    --out_dir="$FILESDIR/models" \
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
