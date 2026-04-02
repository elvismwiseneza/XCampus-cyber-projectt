from __future__ import annotations

import argparse
import json

from scripts.pipeline import PIPELINE_PATHS, generate_raw_logs, preprocess_logs, train_and_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full campus cybersecurity risk monitor pipeline.")
    parser.add_argument("--days", type=int, default=180, help="Number of synthetic days to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible output.")
    args = parser.parse_args()

    print("Step 1/3: Generating synthetic campus security logs...")
    raw_logs = generate_raw_logs(output_path=PIPELINE_PATHS.raw_logs, days=args.days, seed=args.seed)
    print(f"  Created {len(raw_logs)} raw events.")

    print("Step 2/3: Engineering risk features from the raw event stream...")
    cleaned = preprocess_logs(raw_path=PIPELINE_PATHS.raw_logs, output_path=PIPELINE_PATHS.cleaned_logs, seed=args.seed)
    print(f"  Created {len(cleaned)} aggregated feature rows.")

    print("Step 3/3: Training models and producing risk scores...")
    metrics = train_and_score(cleaned_path=PIPELINE_PATHS.cleaned_logs, paths=PIPELINE_PATHS, seed=args.seed)
    metrics["raw_log_rows"] = int(len(raw_logs))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
