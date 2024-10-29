#!/usr/bin/env bash

# Run the gen_test_noise script
poetry run python -m src.gen_test_noise \
    --out_dir="$FILESDIR/out" \
    --seed=42 \
    --nz=10 \
    --z-dim=50
