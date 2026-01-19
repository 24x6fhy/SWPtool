# config loaders

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import yaml
import random
    
def load_weights(config_path: str = None):
    if config_path is None:
        current_file = Path(__file__).resolve()
        config_path = current_file.parent.parent.parent / "configs" / "weights.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find weights configuration at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Success reading weights config from {config_path}")
        return config.get("topic_weights", {})  # here are keys: topic patterns and values


def normalize_weights(weight_map: dict) -> dict:
    normalized = {}
    for key in list(weight_map.keys()):
        try:
            normalized[key] = float(weight_map[key])
        except (ValueError, TypeError):
            print(f"Warning: Skipping invalid weight for '{key}'")
    
    if "default" not in normalized:
        normalized["default"] = 1.0
    
    print(f"Normalized weight map")
    return normalized

def load_and_prepare(csv_path: str): # load features and prep data
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"distance_km"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"features.csv missing columns: {missing}")
    
    good_rows = []
    for idx, row in df.iterrows():
        dist = row["distance_km"]
        if pd.notna(dist) and str(dist) != "N/A" and float(dist) > 0:
            good_rows.append(idx)
    
    df = df.loc[good_rows]

    if df.empty:
        raise RuntimeError(
            "No valid data found in features.csv. "
            "All rows have invalid or missing distance_km values. "
            "Please check that odometry data is available in the ROS bags and re-run feature extraction."
        )

    # target
    if "weighted_pts" not in df.columns:
        raise RuntimeError("features.csv missing 'weighted_pts' column")
    
    df["proxy_total_pts"] = df["weighted_pts"].astype(float)
    df["pts_per_km"] = df["proxy_total_pts"] / df["distance_km"]

    candidate_feats = [
        "duration",
        "distance_km",
        "avg_speed_kmh",
        "image_ratio",
        "lidar_ratio", 
        "radar_ratio",
        "imu_ratio",
        "odometry_ratio",
        "lidar_to_camera_ratio",
        "radar_to_lidar_ratio", 
        "perception_to_nav_ratio",
        "n_active_topics",
    ]

    feature_names = []
    for feat in candidate_feats:
        if feat in df.columns:
            feature_names.append(feat)
    
    if not feature_names:
        raise RuntimeError("No valid feature columns found in features.csv")

    # data fix
    X = df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    y = df["proxy_total_pts"].astype(float)
    groups = df["run_id"] if "run_id" in df.columns else None
    distance_km = df["distance_km"].astype(float).to_numpy()

    return X, y, feature_names, groups, distance_km, df


def split_data(X, y, groups, test_size=0.2, random_state=42):
    if groups is not None and len(groups) > 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return (
            X.iloc[train_idx], 
            X.iloc[test_idx], 
            y.iloc[train_idx], 
            y.iloc[test_idx], 
            train_idx, 
            test_idx
        )
    else:

        # manual split when no groups
        total_rows = len(X)
        test_count = int(total_rows * test_size)
        train_count = total_rows - test_count
        
        # shuffle 
        indices = list(range(total_rows))
        for i in range(len(indices) - 1, 0, -1):
            j = random_state % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        
        train_idx = indices[:train_count]
        test_idx = indices[train_count:]
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx