from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline import PIPELINE_PATHS, preprocess_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw cybersecurity logs into modeling features.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the synthetic incident label.")
    args = parser.parse_args()

    cleaned = preprocess_logs(raw_path=PIPELINE_PATHS.raw_logs, output_path=PIPELINE_PATHS.cleaned_logs, seed=args.seed)
    print(f"Created {len(cleaned)} feature rows at {PIPELINE_PATHS.cleaned_logs}")


if __name__ == "__main__":
    main()
