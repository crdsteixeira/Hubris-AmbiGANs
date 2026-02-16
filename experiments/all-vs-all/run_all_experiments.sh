#!/bin/bash
# Run all digit pair experiments for MNIST or Fashion-MNIST datasets
# Usage: ./run_all_experiments.sh [mnist|fashion-mnist]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR"

# Default to mnist if not specified
DATASET="${1:-mnist}"

# Validate dataset argument
if [[ ! "$DATASET" =~ ^(mnist|fashion-mnist)$ ]]; then
    echo "Error: Invalid dataset '$DATASET'. Must be 'mnist' or 'fashion-mnist'"
    echo "Usage: $0 [mnist|fashion-mnist]"
    exit 1
fi

cd "$PROJECT_ROOT"

# Count total configs for the specified dataset
TOTAL_CONFIGS=$(ls -1 "$CONFIGS_DIR"/${DATASET}-[0-9]v[0-9].yml "$CONFIGS_DIR"/${DATASET}-[0-9]v[0-9][0-9].yml 2>/dev/null | wc -l)
echo "Found $TOTAL_CONFIGS configuration files for $DATASET to run"

if [ "$TOTAL_CONFIGS" -eq 0 ]; then
    echo "Error: No configuration files found for $DATASET"
    echo "Please run: python generate_configs.py --dataset $DATASET"
    exit 1
fi

# Process each config
COUNT=0
for config_file in "$CONFIGS_DIR"/${DATASET}-[0-9]v[0-9].yml "$CONFIGS_DIR"/${DATASET}-[0-9]v[0-9][0-9].yml; do
    if [ ! -f "$config_file" ]; then
        continue
    fi
    
    COUNT=$((COUNT + 1))
    CONFIG_NAME=$(basename "$config_file" .yml)
    echo ""
    echo "=========================================="
    echo "[$COUNT/$TOTAL_CONFIGS] Running: $CONFIG_NAME"
    echo "=========================================="
    
    python -m src --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $CONFIG_NAME"
    else
        echo "✗ Failed: $CONFIG_NAME"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All $DATASET experiments completed successfully!"
echo "=========================================="
