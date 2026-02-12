#!/usr/bin/env python3
"""Generate all-vs-all MNIST digit pair configuration files."""

import re
from pathlib import Path


def generate_all_configs() -> None:
    """Generate YAML config files for all digit pair combinations (0-9 vs all others)."""
    template_path = Path(__file__).parent / "mnist-0v1.yml"

    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        return

    # Read the template
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    # Generate combinations without duplicates: 0v1, 0v2, ..., 0v9, 1v2, ..., 8v9
    generated_count = 0
    for pos in range(10):
        for neg in range(pos + 1, 10):  # Only generate pairs where pos < neg

            # Create new config content
            config_content = template_content

            # Replace name
            config_content = re.sub(r"name: mnist-\d+v\d+", f"name: mnist-{pos}v{neg}", config_content)

            # Replace pos and neg in dataset section
            config_content = re.sub(r"pos: \d+\s+# <<<<<<", f"pos: {pos}   # <<<<<<", config_content)
            config_content = re.sub(r"neg: \d+\s+# <<<<<<", f"neg: {neg}   # <<<<<<", config_content)

            # Replace classifier names
            old_classifier_name = "ensemble_mean_32_mnist_50_0v1_balanced"
            new_classifier_name = f"ensemble_mean_32_mnist_50_{pos}v{neg}_balanced"
            config_content = config_content.replace(old_classifier_name, new_classifier_name)

            # Write the new config file
            output_path = Path(__file__).parent / f"mnist-{pos}v{neg}.yml"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(config_content)

            generated_count += 1
            print(f"Generated: {output_path.name}")

    print(f"\nSuccessfully generated {generated_count} configuration files.")


if __name__ == "__main__":
    generate_all_configs()
