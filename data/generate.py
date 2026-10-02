"""Generate synthetic data for the APO framework experiments."""

import os
import sys

# Ensure the project root is on the path so `src` is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dgp import GaussianAPOSystem

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_e1.csv')


def main():
    system = GaussianAPOSystem()
    df = system.simulate(n=2000, seed=42)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} observations to {OUTPUT_PATH}")
    print("\nSummary statistics:")
    print(df.describe().round(4).to_string())


if __name__ == '__main__':
    main()
