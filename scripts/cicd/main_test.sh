#!/usr/bin/env bash

# Get the directory containing this script (relative to current directory)
SCRIPT_DIR="$(pwd)/scripts/cicd"

# Run the gen_test_noise script
python -m src \
    --config="$SCRIPT_DIR/config_main_test.yml"
