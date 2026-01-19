# RACECAR Energy Proxy

A deterministic proxy metric for comparing sensing workload across autonomous systems using ROS2 logs.

## What Is This?

This tool computes **proxy score** — a surrogate metric that approximates relative sensor load from ROS2 logs **when no power telemetry is available**. It provides a comparable score for ranking or benchmarking different configurations.

**Use cases:**
- Autonomous vehicles without power measurement hardware
- Comparing sensor configurations across different runs
- Benchmarking workload in racing, robotics, or any ROS2-based system

**Proxy formula:**
```
proxy_pts = Σ (message_count × topic_weight)
```

Higher proxy points = higher sensor activity = higher inferred workload.

This tool was developed using the RACECAR dataset but can be applied to **any ROS2 rosbag** by customizing `configs/weights.yaml` for your sensor topics.

---

## Dataset (RACECAR)

This repository includes examples using the **RACECAR Dataset** — the first open dataset for full-scale, high-speed autonomous racing.

**Source:** [linklab-uva/RACECAR_DATA](https://github.com/linklab-uva/RACECAR_DATA)

### Download via AWS CLI (Recommended)

```powershell
# Install AWS CLI: https://aws.amazon.com/cli/

# Download a single scenario (e.g., S1 - Solo Slow Lap)
aws s3 cp s3://racecar-dataset/RACECAR-ROS2/S1/M-SOLO-SLOW-KAIST ./data/S1/M-SOLO-SLOW-KAIST --recursive --no-sign-request



# Download all scenarios
aws s3 cp s3://racecar-dataset/RACECAR-ROS2/ ./data/ --recursive --no-sign-request
```

### Dataset Structure

Place downloaded `.db3` files in the `data/` folder:
```
data/
├── S1/                    # Solo Slow < 70 mph
│   ├── C-SOLO-SLOW-70/
│   ├── M-SOLO-SLOW-70/
│   └── P-SOLO-SLOW-70/
├── S2/                    # Solo Slow 70-100 mph
├── S3/                    # Solo Fast 100-140 mph
├── S4/                    # Solo Fast > 140 mph
├── S5/                    # Multi-Agent Slow
├── S6/                    # Multi-Agent Fast
...
```
---

## Quick Start

```powershell
# Install
git clone https://github.com/24x6fhy/SWPtool.git
cd SWPtool
pip install -r requirements.txt

# Step 1: Compute proxy scores per run
python src/scripts/run_proxy.py
# Output: outputs/proxy_results.csv

# Step 2: Build features 
python src/scripts/extract_features.py
# Output: outputs/features.csv

# Step 3: Train ML models for robustness analysis
python src/scripts/train_model.py
# Output: outputs/models/
```

**Requirements:** Python 3.10+, pandas, numpy, scikit-learn, xgboost, matplotlib, PyYAML

---

## CLI Options

All scripts support customization via command-line arguments:

### `run_proxy.py` — Compute proxy scores

```powershell
python src/scripts/run_proxy.py --help

# Custom data folder and weights
python src/scripts/run_proxy.py --data /path/to/rosbags --weights ./my_weights.yaml

# Custom odometry topic for distance calculation
python src/scripts/run_proxy.py --odometry-topic /vehicle/odom --output ./results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data`, `-d` | `./data` | Root folder containing `.db3` files |
| `--weights`, `-w` | `./configs/weights.yaml` | Topic weights YAML |
| `--output`, `-o` | `./outputs` | Output folder for results |
| `--odometry-topic` | `local_odometry` | Odometry topic substring |
| `--exclude` | none | Folders to skip (e.g., `--exclude S1 S2`) |

### `extract_features.py` — Build ML features

```powershell
python src/scripts/extract_features.py --help

# Custom slice duration
python src/scripts/extract_features.py --slice 30 --data ./my_data

# Exclude specific scenarios
python src/scripts/extract_features.py --exclude S5 S6
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data`, `-d` | `./data` | Root folder containing `.db3` files |
| `--output`, `-o` | `./outputs/features.csv` | Output CSV path |
| `--slice`, `-s` | `60` | Time slice duration (seconds) |
| `--weights`, `-w` | `./configs/weights.yaml` | Topic weights YAML |
| `--odometry-topic` | `local_odometry` | Odometry topic substring |
| `--exclude` | none | Folders to skip |

### `train_model.py` — Train ML models

```powershell
python src/scripts/train_model.py --help

# Train only specific models
python src/scripts/train_model.py --models random_forest xgboost

# Fast mode (skip plots and robustness)
python src/scripts/train_model.py --skip-plots --skip-robustness

# Custom hyperparameters
python src/scripts/train_model.py --n-estimators 500 --max-depth 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--features`, `-f` | auto-search | Path to `features.csv` |
| `--output`, `-o` | `./outputs/models` | Output folder for models |
| `--models`, `-m` | all | Models to train (`linear_regression`, `random_forest`, `xgboost`) |
| `--test-size` | `0.2` | Test split fraction |
| `--random-state` | `42` | Random seed |
| `--n-estimators` | `300` | Trees for RF/XGB |
| `--max-depth` | `6` | Max depth for XGB |
| `--skip-plots` | false | Skip generating plots |
| `--skip-robustness` | false | Skip robustness testing |

---

## How It Works

### Deterministic Proxy

1. **Read ROS logs** (`.db3` files)
2. **Count messages** per sensor topic
3. **Apply weights** based on sensor type (lidar > camera > IMU)
4. **Sum weighted counts** → proxy score (Ptotal)
5. **Normalize by distance** → pts_per_km (Pkm)

**Example weights** (`configs/weights.yaml`):
| Sensor | Weight | Rationale |
|--------|--------|-----------|
| Lidar | 10.0 | High bandwidth |
| Camera | 3.0 | Moderate bandwidth |
| Radar | 3.0 | Moderate bandwidth |
| IMU | 0.5 | Low bandwidth |
| Odometry | 0.1 | Minimal |

### ML Models

The deterministic proxy is transparent but may fail when logs are incomplete (missing topics, corrupted modality streams). ML models are trained to predict the proxy from aggregated features.

**Features** (11 total — see paper Section 3.2.5):
- **Temporal**: Duration of time slice (seconds)
- **Spatial**: Distance traveled (km), average speed (km/h)
- **Cross-sensor ratios**: lidar-to-camera, radar-to-lidar, perception-to-navigation
- **Per-modality ratios**: fraction of each sensor's contribution (image, lidar, radar, IMU, odometry)

Note: Raw message counts and per-sensor rates are excluded from features to avoid circular correlation with the target (proxy is computed from counts).

**Target**: Total proxy points (Ptotal = Σ message_count × weight)

**Models**:
| Model | Purpose |
|-------|---------|
| Linear Regression | Interpretable baseline, captures linear relationships |
| Random Forest | Nonlinear interactions, feature importance scores |
| XGBoost | Gradient-boosted ensemble for tabular data |

**Data augmentation**: Each run is divided into fixed-length segments (default: 60s). Segments are treated as independent samples for training.

**Group-based split**: Segments from the same run never appear in both train and test sets.

### Robustness Experiments (Feature Dropout)

To test proxy robustness under partial observability, sensor modalities are simulated as missing by zeroing out their features:

| Scenario | Features Dropped |
|----------|------------------|
| `no_vision` | `image_rate`, `image_ratio` |
| `no_lidar` | `lidar_rate`, `lidar_ratio` |
| `no_radar` | `radar_rate`, `radar_ratio` |
| `no_imu` | `imu_rate`, `imu_ratio` |
| `no_odometry` | `odometry_rate`, `odometry_ratio` |

---

## Outputs

### `outputs/proxy_results.csv` (from `run_proxy.py`)

Per-database proxy scores:
```csv
database_name,weighted_msg_count,distance_km,pts_per_hour,pts_per_km
C-SOLO-SLOW-70,125430.5,2.15,45230.2,58340.5
M-SOLO-SLOW-70,98320.1,1.89,38120.8,52021.7
```

| Column | Description |
|--------|-------------|
| `weighted_msg_count` | Total proxy score for entire run |
| `pts_per_hour` | Proxy normalized by duration |
| `pts_per_km` | Proxy normalized by distance |

### `outputs/features.csv` (from `extract_features.py`)

Per-slice features for ML:
```csv
bag_name,slice_idx,duration,distance_km,weighted_pts,pts_per_km,...
C-SOLO-SLOW-70,0,60.0,0.12,1425.5,11879.2,...
C-SOLO-SLOW-70,1,60.0,0.14,1568.3,11202.1,...
```

| Column | Description |
|--------|-------------|
| `weighted_pts` | Proxy score for fixed-length window |
| `pts_per_km` | Proxy normalized by distance |
| `image_rate`, `lidar_rate`, ... | Sensor activity features |

### `outputs/models/` (from `train_model.py`)

```
outputs/models/
├── linear_regression.joblib
├── random_forest.joblib
├── xgboost.joblib
├── evaluation_metrics.json
├── robustness_feature_dropout.csv   ← Sensor failure impact
└── plots/
    ├── model_comparison.png
    └── feature_importance.png
```

---

## Key Outputs

### `robustness_feature_dropout.csv`

Shows model performance degradation when each sensor modality is missing:

```csv
model,scenario,r2,rmse,r2_degradation,rmse_increase_pct
random_forest,no_vision,0.982,7200.5,0.003,9.8
random_forest,no_lidar,0.975,8100.2,0.010,23.5
random_forest,no_radar,0.950,11500.1,0.035,75.2
...
```

### `training_summary.json`

Complete training configuration and results for reproducibility.

---

## Configuration

Edit `configs/weights.yaml` to adjust topic mapping/ weights for your use case.

---

## Project Structure

```
src/
├── features/        Extract features from .db3 files
├── proxy/           Compute weighted proxy scores
├── ml/              ML models for robustness testing
└── scripts/         CLI tools (run_proxy.py, extract_features.py, train_model.py)

configs/             Weight configuration
data/                ROS .db3 database files
outputs/             Results (features.csv, models/)
```

---

## License

MIT License
