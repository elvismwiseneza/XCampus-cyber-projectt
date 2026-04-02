from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline import PIPELINE_PATHS, train_and_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the campus risk monitor and generate deployment artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible model training.")
    args = parser.parse_args()

    metrics = train_and_score(cleaned_path=PIPELINE_PATHS.cleaned_logs, paths=PIPELINE_PATHS, seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
