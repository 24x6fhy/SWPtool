# compute proxy for all bags

import csv
import json
from pathlib import Path
from datetime import datetime

from .deterministic import weighted_msg_count, simple_msg_count, get_drive_duration
from .odometry import get_distance_km_from_topic


def process_one_bag(db_path, weights, odom_topic="local_odometry"):

    km = get_distance_km_from_topic(str(db_path), topic_name_substring=odom_topic, verbose=False)
    
    secs = get_drive_duration(str(db_path), time_unit="seconds")
    hrs = secs / 3600
    
    by_topic = weighted_msg_count(str(db_path), weights)
    total_pts = 0
    for topic in by_topic:
        total_pts += by_topic[topic]
    
    raw_count = simple_msg_count(str(db_path))
    
    # rates
    pts_hr = 0
    if hrs > 0:
        pts_hr = total_pts / hrs
    
    pts_km = None
    if km is not None and km > 0:
        pts_km = total_pts / km
    
    return {
        "database_name": db_path.stem,
        "database_path": str(db_path),
        "simple_msg_count": raw_count,
        "weighted_msg_count": round(total_pts, 2),
        "duration_hours": round(hrs, 4),
        "duration_seconds": int(secs),
        "distance_km": round(km, 2) if km else None,
        "pts_per_hour": round(pts_hr, 2),
        "pts_per_km": round(pts_km, 2) if pts_km else None,
    }


def sum_proxy(db_files, weights, output_dir, odometry_topic="local_odometry", config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("PROXY COMPUTATION")
    
    results = []
    for bag in db_files:
        try:
            r = process_one_bag(bag, weights, odometry_topic)
            results.append(r)
            print(f"  Processed:", bag.name)
        except:
            pass  # skip bad ones
    
    if len(results) == 0:
        return None
    
    f = open(output_dir / "proxy_results.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    f.close()
    
    total_pts = 0
    total_hrs = 0
    total_km = 0
    total_msgs = 0
    
    for r in results:
        total_pts += r["weighted_msg_count"]
        total_hrs += r["duration_hours"]
        total_msgs += r["simple_msg_count"]
        if r["distance_km"] is not None:
            total_km += r["distance_km"]
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config if config else {},
        "databases_processed": len(results),
        "databases_found": len(db_files),
        "total_messages": total_msgs,
        "total_weighted_pts": round(total_pts, 2),
        "total_duration_hours": round(total_hrs, 2),
        "total_distance_km": round(total_km, 2),
    }
    
    if total_hrs > 0:
        summary["avg_pts_per_hour"] = round(total_pts / total_hrs, 2)
    else:
        summary["avg_pts_per_hour"] = 0
        
    if total_km > 0:
        summary["avg_pts_per_km"] = round(total_pts / total_km, 2)
    else:
        summary["avg_pts_per_km"] = 0
    
    f = open(output_dir / "proxy_summary.json", "w")
    json.dump(summary, f, indent=2)
    f.close()
    
    return summary
