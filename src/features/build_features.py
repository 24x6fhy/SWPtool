# main feature pipeline

import sqlite3
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .extractors import extract_features
from utils.loaders import load_weights, normalize_weights
from utils.db_utils import find_all_db3_files


def extract_all_features(
    root_path=None,
    output_csv="features.csv",
    slice_seconds=60,
    exclude_patterns=None,
    weights_path=None,
    odometry_topic="local_odometry",
):
    if root_path is None:
        root_path = Path(".")
    else:
        root_path = Path(root_path)
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    print("FEATURE EXTRACTION")
    
    weight_map = load_weights(weights_path)
    weight_map = normalize_weights(weight_map)
    
    print("Searching for .db3 files...")
    db_files = find_all_db3_files(root_path=root_path, exclude_patterns=exclude_patterns)
    print(f"Found {len(db_files)} database files")
    
    if not db_files:
        print("No database files found!")
        return
    
    all_rows = []
    slice_ns = int(slice_seconds * 1e9)
    count = 0

    for db3_file in sorted(db_files):
        count += 1
        print(f"[{count}/{len(db_files)}] Processing {db3_file.name}...")
        
        try:
            db = sqlite3.connect(str(db3_file))
            cursor = db.cursor()
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages")
            t_min, t_max = cursor.fetchone()
            db.close()
        except Exception as e:
            print(f"   Error reading database: {e}")
            continue
        
        if t_min is None or t_max is None:
            print(f"   No messages found, skipping")
            continue
        
        current = int(t_min)
        t_max_i = int(t_max)
        idx = 0
        bag_name = db3_file.stem
        slice_count = 0
        
        while current < t_max_i:
            row = extract_features(
                str(db3_file),
                current,
                current + slice_ns,
                bag_name,
                idx,
                weight_map,
                odometry_topic
            )
            if row:
                all_rows.append(row)
                slice_count += 1
            current += slice_ns
            idx += 1
        
        print(f"   Extracted {slice_count} slices")
    
    print("SAVING RESULTS")
    
    if all_rows:
        output_path = Path(output_csv)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} feature rows to: {output_path}")
    else:
        print("No valid data extracted.")
    
    return all_rows
