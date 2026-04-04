from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline import PIPELINE_PATHS, generate_raw_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic campus cybersecurity event logs.")
    parser.add_argument("--days", type=int, default=180, help="Number of days of synthetic data to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible output.")
    args = parser.parse_args()
    
    logs = generate_raw_logs(output_path=PIPELINE_PATHS.raw_logs, days=args.days, seed=args.seed)
    print(f"Generated {len(logs)} raw events at {PIPELINE_PATHS.raw_logs}")


if __name__ == "__main__":
    main()
