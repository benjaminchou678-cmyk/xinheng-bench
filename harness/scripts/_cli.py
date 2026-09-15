import argparse
import sys
from pathlib import Path

# Supports direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import load_config


def main(action):
    parser = argparse.ArgumentParser(description="Chinese herbal medicine bias " + action)
    parser.add_argument("--config", type=Path, help="Benchmark YAML; paths inside it are project-root relative.")
    parser.add_argument("--run-id", required=True, help="Unique run id; use the same id for all three stages.")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if action == "generation":
            from pipeline.generate import run
        elif action == "evaluation":
            from pipeline.evaluate import run
        else:
            from pipeline.aggregate import run
        print(run(config, args.run_id))
    except Exception as exc:
        parser.exit(1, f"{action} failed: {exc}\n")
