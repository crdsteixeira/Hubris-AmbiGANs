#!/usr/bin/env bash

# Resolve script directory
SCRIPT_DIR="$(dirname "$0")"

# Run the gen_test_noise script
python -m src \
    --config="$SCRIPT_DIR/config_main_test.yml"
