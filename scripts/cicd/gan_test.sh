#!/usr/bin/env bash

# Get the directory containing this script (relative to current directory)
SCRIPT_DIR="$(pwd)/scripts/cicd"

# Run the gen_test_noise script
python -m src.gan.gan_cli \
    --config="$SCRIPT_DIR/config_gan_test.yml"
