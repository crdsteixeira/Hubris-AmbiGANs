#!/usr/bin/env bash

# Resolve project root from script location (absolute path)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Run the gen_test_noise script
python -m src.gen_test_noise \
    --out-dir="$ROOT_DIR/data" \
    --seed=42 \
    --nz=10 \
    --z-dim=10
