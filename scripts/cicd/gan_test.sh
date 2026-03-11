#!/usr/bin/env bash

# Resolve script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# Run the gen_test_noise script
python -m src.gan.gan_cli \
    --config="$SCRIPT_DIR/config_gan_test.yml"
