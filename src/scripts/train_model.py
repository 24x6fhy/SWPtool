#!/usr/bin/env python
# command line interface for training ML models

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

from utils.loaders import load_and_prepare
from ml import run_pipeline


def find_features():
    locations = [
        Path.cwd() / "features.csv",
        Path(__file__).parent.parent.parent / "outputs" / "features.csv",
    ]
    for loc in locations:
        if loc.exists():
            return loc
    return None


def main():
    parser = argparse.ArgumentParser(description="Train ML models for proxy prediction")
    parser.add_argument("--features", "-f", type=str, help="Path to features.csv")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", "-m", nargs="+", default=["linear_regression", "random_forest", "xgboost"],
                        choices=["linear_regression", "random_forest", "xgboost"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    
    features_path = Path(args.features) if args.features else find_features()
    if not features_path or not features_path.exists():
        print("Error: features.csv not found. Run extract_features.py first.")
        return
    
    output_dir = Path(args.output) if args.output else Path(__file__).parent.parent.parent / "outputs" / "models"
    plots_dir = output_dir / "plots"
    proxy_results_path = features_path.parent / "proxy_results.csv"
    
    print("[1/4] Loading data...")
    X, y, feat_list, groups, distance_km, df = load_and_prepare(features_path)
    print(f"      {len(y)} samples, {len(feat_list)} features")

    print("[2/4] Training...")
    summary = run_pipeline(
        X, y, df, feat_list, groups,
        output_dir=output_dir,
        plots_dir=plots_dir,
        models=args.models,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        skip_plots=args.skip_plots,
        skip_robustness=args.skip_robustness,
        proxy_results_path=proxy_results_path,
        features_path=features_path,
        verbose=False,
    )
    
    if not summary:
        print("No models trained")
        return
    
    print("[3/4] Results:")
    for m in summary["models"]:
        print(f"      {m['name']:<18} R²={m['r2']:.4f}")
    print(f"[4/4] Saved to {output_dir}/")
    print(f"\nBest: {summary['best_model']} (R²={summary['best_r2']:.4f})")


if __name__ == "__main__":
    main()
