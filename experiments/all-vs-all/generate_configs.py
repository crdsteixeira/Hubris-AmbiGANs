#!/usr/bin/env python3
"""Generate all-vs-all digit pair configuration files for MNIST or Fashion-MNIST datasets."""

import argparse
import re
import sys
from pathlib import Path


def generate_all_configs(dataset: str) -> None:
    """
    Generate YAML config files for all digit pair combinations (0-9 vs all others).

    Args:
        dataset: Either 'mnist' or 'fashion-mnist'

    """
    # Validate dataset
    if dataset not in ["mnist", "fashion-mnist"]:
        print(f"Error: Invalid dataset '{dataset}'. Must be 'mnist' or 'fashion-mnist'")
        sys.exit(1)

    # Look for template in ambigan subdirectory first, then fallback to current directory
    template_filename = f"{dataset}-0v1.yml"
    template_path = Path(__file__).parent / "ambigan" / template_filename

    if not template_path.exists():
        template_path = Path(__file__).parent / template_filename

    if not template_path.exists():
        print("Error: Template file not found. Checked:")
        print(f"  - {Path(__file__).parent / 'ambigan' / template_filename}")
        print(f"  - {Path(__file__).parent / template_filename}")
        sys.exit(1)

    # Read the template
    with open(template_path, encoding="utf-8") as f:
        template_content = f.read()

    # Generate combinations without duplicates: 0v1, 0v2, ..., 0v9, 1v2, ..., 8v9
    generated_count = 0
    for pos in range(10):
        for neg in range(pos + 1, 10):  # Only generate pairs where pos < neg

            # Create new config content
            config_content = template_content

            # Replace name to use dataset prefix
            config_content = re.sub(rf"name: {dataset}-\d+v\d+", f"name: {dataset}-{pos}v{neg}", config_content)

            # Replace pos and neg in dataset section
            config_content = re.sub(r"pos: \d+\s+# <<<<<<", f"pos: {pos}   # <<<<<<", config_content)
            config_content = re.sub(r"neg: \d+\s+# <<<<<<", f"neg: {neg}   # <<<<<<", config_content)

            # Replace classifier names
            old_classifier_name = f"ensemble_mean_32_{dataset}_50_0v1_balanced"
            new_classifier_name = f"ensemble_mean_32_{dataset}_50_{pos}v{neg}_balanced"
            config_content = config_content.replace(old_classifier_name, new_classifier_name)

            # Write the new config file
            output_path = Path(__file__).parent / f"{dataset}-{pos}v{neg}.yml"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(config_content)

            generated_count += 1
            print(f"Generated: {output_path.name}")

    print(f"\nSuccessfully generated {generated_count} configuration files for {dataset}.")


def main() -> None:
    """Parse arguments and generate configs."""
    parser = argparse.ArgumentParser(description="Generate all-vs-all digit pair configuration files")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion-mnist"],
        help="Dataset to generate configs for (default: mnist)",
    )

    args = parser.parse_args()
    generate_all_configs(args.dataset)


if __name__ == "__main__":
    main()
