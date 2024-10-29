#!/usr/bin/env bash

# Run the gen_test_noise script
poetry run python -m src.gan.gan_cli \
    --config="./scripts/cicd/config_gan_test.yml"
