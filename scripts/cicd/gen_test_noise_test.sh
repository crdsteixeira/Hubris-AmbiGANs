#!/usr/bin/env bash

# Resolve project root from script location
SCRIPT_DIR="$(dirname "$0")"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Run the gen_test_noise script
python -m src.gen_test_noise \
    --out-dir="$ROOT_DIR/data" \
    --seed=42 \
    --nz=10 \
    --z-dim=10
