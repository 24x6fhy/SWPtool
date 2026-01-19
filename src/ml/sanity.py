# sanity checks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_sanity_checks(proxy_results_path, plots_dir, features_path=None):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    df = pd.read_csv(proxy_results_path)
    df = df[df["duration_seconds"] > 0]
    df = df[df["distance_km"].notna() & (df["distance_km"] > 0)]
    df["avg_speed_kmh"] = df["distance_km"] / df["duration_hours"]
    df["avg_speed_kmh"] = df["avg_speed_kmh"].replace([np.inf, -np.inf], np.nan)
    
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # distance vs SWP score
    ax = axes[0]
    ax.scatter(df["distance_km"], df["weighted_msg_count"], alpha=0.7, s=80, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Distance (km)', fontweight='bold')
    ax.set_ylabel('Total Proxy Points', fontweight='bold')
    ax.set_title('SC1: Distance vs Total SWP', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    r1 = df[["distance_km", "weighted_msg_count"]].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'r = {r1:.3f}', transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    results.append({"check": "SC1", "x": "Distance", "y": "Total SWP", "tests": "Accumulation", 
                    "level": "run", "n": len(df), "r": round(r1, 4)})
    
    # duration vs SWP/hr
    ax = axes[1]
    ax.scatter(df["duration_seconds"], df["pts_per_hour"], alpha=0.7, s=80, edgecolors='k', 
               linewidth=0.5, color='green')
    ax.set_xlabel('Duration (s)', fontweight='bold')
    ax.set_ylabel('SWP/hr', fontweight='bold')
    ax.set_title('SC2: Duration vs SWP/hr', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    r2 = df[["duration_seconds", "pts_per_hour"]].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'r = {r2:.3f}', transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    results.append({"check": "SC2", "x": "Duration", "y": "SWP/hr", "tests": "Time norm", 
                    "level": "run", "n": len(df), "r": round(r2, 4)})
    
    # speed vs PWS/km
    ax = axes[2]
    valid = df[(df["avg_speed_kmh"].notna()) & (df["avg_speed_kmh"] > 0)]
    ax.scatter(valid["avg_speed_kmh"], valid["pts_per_km"], alpha=0.7, s=80, edgecolors='k', 
               linewidth=0.5, color='red')
    ax.set_xlabel('Speed (km/h)', fontweight='bold')
    ax.set_ylabel('SWP/km', fontweight='bold')
    ax.set_title('SC3: Speed vs SWP/km', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    r3 = valid[["avg_speed_kmh", "pts_per_km"]].corr().iloc[0, 1] if len(valid) > 1 else np.nan
    if not np.isnan(r3):
        ax.text(0.05, 0.95, f'r = {r3:.3f}', transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    results.append({"check": "SC3", "x": "Speed", "y": "SWP/km", "tests": "Distance norm", 
                    "level": "run", "n": len(valid), "r": round(r3, 4) if not np.isnan(r3) else np.nan})
    
    plt.tight_layout()
    plot_path = plots_dir / "proxy_sanity_checks.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # slice-level speed vs SWP/km
    r4, n4 = np.nan, 0
    if features_path and Path(features_path).exists():
        slices = pd.read_csv(features_path)
        slices = slices[slices["distance_km"].notna() & (slices["distance_km"] > 0)]
        slices["speed"] = slices["distance_km"] / (slices["duration"] / 3600)
        slices["speed"] = slices["speed"].replace([np.inf, -np.inf], np.nan)
        slices["pts_km"] = slices["weighted_pts"] / slices["distance_km"]
        
        valid_s = slices[(slices["speed"].notna()) & (slices["speed"] > 0) & (slices["pts_km"].notna())]
        n4 = len(valid_s)
        if n4 > 1:
            r4 = valid_s[["speed", "pts_km"]].corr().iloc[0, 1]
    
    results.append({"check": "SC4", "x": "Speed", "y": "SWP/km", "tests": "Distance norm", 
                    "level": "slice", "n": n4, "r": round(r4, 4) if not np.isnan(r4) else np.nan})
    
    summary = pd.DataFrame(results)
    summary_path = plots_dir.parent / "sanity_check_summary.csv"
    summary.to_csv(summary_path, index=False)
    
    return plot_path, summary
