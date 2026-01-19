# Robustness tests

from sklearn.metrics import r2_score, mean_squared_error


def test_feature_dropout(model, X_test, y_test, feature_names, model_name):
    baseline = model.predict(X_test)
    r2_ok = r2_score(y_test, baseline)
    rmse_ok = mean_squared_error(y_test, baseline) ** 0.5
    
    out = []
    
    scenarios = [
        ("no_vision", ["image_ratio", "lidar_to_camera_ratio"]),
        ("no_lidar", ["lidar_ratio", "lidar_to_camera_ratio", "radar_to_lidar_ratio"]),
        ("no_radar", ["radar_ratio", "radar_to_lidar_ratio"]),
        ("no_imu", ["imu_ratio", "perception_to_nav_ratio"]),
        ("no_odometry", ["odometry_ratio", "perception_to_nav_ratio", "avg_speed_kmh", "distance_km"]),
    ]
    
    for name, feats in scenarios:
        X_broken = X_test.copy()
        
        for f in feats:
            if f in X_broken.columns:
                X_broken[f] = 0.0
        
        pred = model.predict(X_broken)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        
        # degradation
        r2_drop = r2_ok - r2
        if rmse_ok > 0:
            rmse_jump = (rmse - rmse_ok) / rmse_ok * 100
        else:
            rmse_jump = 0
        
        out.append({
            "scenario": name,
            "r2": r2,
            "rmse": rmse,
            "r2_degradation": r2_drop,
            "rmse_increase_pct": rmse_jump,
        })
            
    return out
