#!/usr/bin/env bash

# Get the project root (current directory when running from container)
ROOT_DIR="$(pwd)"

# Run the gen_test_noise script
python -m src.gen_test_noise \
    --out-dir="$ROOT_DIR/data" \
    --seed=42 \
    --nz=10 \
    --z-dim=10
