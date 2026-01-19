# core functions extraction

import sqlite3
import json
import re
import math
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proxy.deterministic import weighted_msg_count
from proxy.odometry import (
    get_distance_km_from_topic,
)


def extract_features(db_path, start_ns, end_ns, bag_name, slice_idx, weight_map, odometry_topic="local_odometry"):
    # exrtact features for one time slice

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM topics")
    topics = dict(cur.fetchall())

    cur.execute(
        "SELECT topic_id, COUNT(*) FROM messages WHERE timestamp >= ? AND timestamp < ? GROUP BY topic_id",
        (int(start_ns), int(end_ns)),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None

    duration = (end_ns - start_ns) / 1e9
    if duration <= 0:
        return None

    topic_counts = {topics.get(tid, "<unknown>"): cnt for tid, cnt in rows}
    total_msgs = sum(topic_counts.values())
    image_msgs = sum(c for t, c in topic_counts.items() if "image" in t.lower())
    lidar_msgs = sum(c for t, c in topic_counts.items() if "luminar" in t.lower() or "lidar" in t.lower())
    radar_msgs = sum(c for t, c in topic_counts.items() if "radar" in t.lower())
    imu_msgs = sum(c for t, c in topic_counts.items() if "imu" in t.lower())
    odometry_msgs = sum(c for t, c in topic_counts.items() if "odometry" in t.lower())

    n_active_topics = len([c for c in topic_counts.values() if c > 0])

    msg_rate = total_msgs / duration if duration > 0 else 0
    image_rate = image_msgs / duration if duration > 0 else 0
    lidar_rate = lidar_msgs / duration if duration > 0 else 0
    radar_rate = radar_msgs / duration if duration > 0 else 0
    imu_rate = imu_msgs / duration if duration > 0 else 0
    odometry_rate = odometry_msgs / duration if duration > 0 else 0
    
    image_ratio = image_msgs / total_msgs if total_msgs > 0 else 0
    lidar_ratio = lidar_msgs / total_msgs if total_msgs > 0 else 0
    radar_ratio = radar_msgs / total_msgs if total_msgs > 0 else 0
    imu_ratio = imu_msgs / total_msgs if total_msgs > 0 else 0
    odometry_ratio = odometry_msgs / total_msgs if total_msgs > 0 else 0

    perception_msgs = image_msgs + lidar_msgs + radar_msgs
    navigation_msgs = imu_msgs + odometry_msgs
    lidar_to_camera_ratio = lidar_msgs / image_msgs if image_msgs > 0 else 0
    radar_to_lidar_ratio = radar_msgs / lidar_msgs if lidar_msgs > 0 else 0
    perception_to_nav_ratio = perception_msgs / navigation_msgs if navigation_msgs > 0 else 0

    weighted_counts = weighted_msg_count(db_path, weight_map, start_ns=start_ns, end_ns=end_ns)
    total_weighted_pts = sum(weighted_counts.values())
    distance_km = get_distance_km_from_topic(db_path, topic_name_substring=odometry_topic, start_ns=start_ns, end_ns=end_ns)
    
    duration_hours = duration / 3600
    avg_speed_kmh = distance_km / duration_hours if (distance_km and duration_hours > 0) else 0
    
    pts_per_km = (total_weighted_pts / distance_km) if distance_km and distance_km > 0 else "N/A"

    return {
        "run_id": bag_name,
        "bag_name": f"{bag_name}_slice{slice_idx}",
        "slice_idx": slice_idx,
        
        "duration": duration,
        "distance_km": distance_km if distance_km is not None else "N/A",
        
        "weighted_pts": total_weighted_pts,
        
        "pts_per_km": pts_per_km,
        
        "total_msgs": total_msgs,
        "image_msgs": image_msgs,
        "lidar_msgs": lidar_msgs,
        "radar_msgs": radar_msgs,
        "imu_msgs": imu_msgs,
        "odometry_msgs": odometry_msgs,
        
        "msg_rate": msg_rate,
        "image_rate": image_rate,
        "lidar_rate": lidar_rate,
        "radar_rate": radar_rate,
        "imu_rate": imu_rate,
        "odometry_rate": odometry_rate,
        
        "image_ratio": image_ratio,
        "lidar_ratio": lidar_ratio,
        "radar_ratio": radar_ratio,
        "imu_ratio": imu_ratio,
        "odometry_ratio": odometry_ratio,
        
        "lidar_to_camera_ratio": lidar_to_camera_ratio,
        "radar_to_lidar_ratio": radar_to_lidar_ratio,
        "perception_to_nav_ratio": perception_to_nav_ratio,
        
        "avg_speed_kmh": avg_speed_kmh,
        
        "n_active_topics": n_active_topics,
    }
