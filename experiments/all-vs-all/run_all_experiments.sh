#!/bin/bash
# Run all MNIST digit pair experiments (non-repeated pairs only: 0v1, 0v2, ..., 8v9)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR"

cd "$PROJECT_ROOT"

# Count total configs
TOTAL_CONFIGS=$(ls -1 "$CONFIGS_DIR"/mnist-[0-9]v[0-9].yml "$CONFIGS_DIR"/mnist-[0-9]v[0-9][0-9].yml 2>/dev/null | wc -l)
echo "Found $TOTAL_CONFIGS configuration files to run"

# Process each config
COUNT=0
for config_file in "$CONFIGS_DIR"/mnist-[0-9]v[0-9].yml "$CONFIGS_DIR"/mnist-[0-9]v[0-9][0-9].yml; do
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
echo "All experiments completed successfully!"
echo "=========================================="
