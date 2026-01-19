#!/usr/bin/env python
# command line interface for running proxy 

import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.db_utils import find_all_db3_files
from utils.loaders import load_weights
from proxy.compute import sum_proxy


def main():
    project_root = Path(__file__).parent.parent.parent
    
    parser = argparse.ArgumentParser(description="Compute proxy scores for ROS2 databases")
    parser.add_argument("--data", "-d", default=str(project_root / "data"))
    parser.add_argument("--weights", "-w", default=str(project_root / "configs" / "weights.yaml"))
    parser.add_argument("--output", "-o", default=str(project_root / "outputs"))
    parser.add_argument("--odometry-topic", default="local_odometry")
    parser.add_argument("--exclude", nargs="+", default=[])
    args = parser.parse_args()
    
    print("[1/3] Discovering databases...")
    db_files = find_all_db3_files(args.data, exclude_patterns=args.exclude)
    if not db_files:
        print("      No .db3 files found")
        return
    print(f"      Found {len(db_files)} databases")
    
    try:
        weights = load_weights(args.weights)
    except Exception as e:
        print(f"      Failed to load weights: {e}")
        return

    print("[2/3] Processing... (might take a little while)")
    config = {"data_path": args.data, "weights_path": args.weights, "odometry_topic": args.odometry_topic}
    summary = sum_proxy(db_files, weights, args.output, args.odometry_topic, config)
    
    if not summary:
        print("      No databases processed")
        return
    
    print(f"[3/3] Done: {summary['databases_processed']}/{summary['databases_found']} databases")
    print(f"      Saved to {args.output}/")


if __name__ == "__main__":
    main()