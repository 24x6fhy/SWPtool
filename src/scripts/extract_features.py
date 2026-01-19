#!/usr/bin/env python
# command line interface for extracting features from ROS2 bags

import argparse
import sys
from pathlib import Path

# parent path for import
sys.path.insert(0, str(Path(__file__).parent.parent))

from features import extract_all_features


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    parser = argparse.ArgumentParser(
        description="Extract ML features from ROS2 databases")
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=str(project_root / "data"),
        help="Root directory containing .db3 files (default: ./data)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(project_root / "outputs" / "features.csv"),
        help="Output CSV file path (default: ./outputs/features.csv)"
    )
    parser.add_argument(
        "--slice", "-s",
        type=int,
        default=60,
        help="Time slice duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=str(project_root / "configs" / "weights.yaml"),
        help="Path to weights YAML file (default: ./configs/weights.yaml)"
    )
    parser.add_argument(
        "--odometry-topic",
        type=str,
        default="local_odometry",
        help="Odometry topic name substring for distance (default: local_odometry)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Folder patterns to exclude (e.g., --exclude S1 S2)"
    )
    
    args = parser.parse_args()
    
    extract_all_features(
        root_path=args.data,
        output_csv=args.output,
        slice_seconds=args.slice,
        weights_path=args.weights,
        exclude_patterns=args.exclude,
        odometry_topic=args.odometry_topic,
    )


if __name__ == "__main__":
    main()
