#!/usr/bin/env python3
"""UCI protocol engine for the fly chess engine to be used with Lichess bot."""

import os
import sys
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)

from src.engines.fly_engine import UciFlyEngine


def main():
    """Start the fly engine in UCI mode for Lichess bot."""
    config_path = os.path.join(project_root, "configs", "engine.yaml")
    model_path = None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_path = config.get("fly_engine", {}).get("model_path")
    except (FileNotFoundError, yaml.YAMLError, AttributeError):
        pass

    if model_path:
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, using default model")
            model_path = None

    engine = UciFlyEngine(model_path)
    engine.run()


if __name__ == "__main__":
    main()
