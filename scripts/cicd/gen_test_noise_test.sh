#!/usr/bin/env bash

# Run the gen_test_noise script
poetry run python -m src.gen_test_noise \
    --out-dir="$FILESDIR/data" \
    --seed=42 \
    --nz=10 \
    --z-dim=10
