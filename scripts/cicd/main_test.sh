#!/usr/bin/env bash

# Run the gen_test_noise script
poetry run python -m src \
    --config="./scripts/cicd/config_main_test.yml"
