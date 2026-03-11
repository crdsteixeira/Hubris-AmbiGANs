#!/usr/bin/env bash

# Resolve project root from script location
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Change to project root directory
cd "$ROOT_DIR" || exit 1

# Run the gen_test_noise script
python -m src.gen_test_noise \
    --out-dir="$ROOT_DIR/data" \
    --seed=42 \
    --nz=10 \
    --z-dim=10
