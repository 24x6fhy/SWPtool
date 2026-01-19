# ML package

import json
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from .train import train_and_evaluate, get_best_model
from .save import save_model, save_importances, save_importance_summary, save_robustness_results
from .robustness import test_feature_dropout
from .plots import plot_feature_importance, plot_predictions, plot_robustness
from .sanity import plot_sanity_checks


def run_pipeline(X, y, df, features, groups, output_dir, plots_dir,
                 models=None, test_size=0.2, random_state=42,
                 n_estimators=300, max_depth=6,
                 skip_plots=False, skip_robustness=False,
                 proxy_results_path=None, features_path=None,
                 verbose=True):
    
    from utils.loaders import split_data
    
    # setup folders
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    if models is None:
        models = ["linear_regression", "random_forest", "xgboost"]
    
    # sanity checks
    if not skip_plots and proxy_results_path:
        if Path(proxy_results_path).exists():
            if verbose: print("  Sanity checks...")
            plot_sanity_checks(proxy_results_path, plots_dir, features_path=features_path)
    
    # split
    if verbose: print(f"  Splitting data...")
    split = split_data(X, y, groups, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = split[0], split[1], split[2], split[3]
    
    results = []
    rob = {}
    
    if "linear_regression" in models:
        if verbose: print("  Training linear_regression...")
        m = LinearRegression()
        r = train_and_evaluate(m, "linear_regression", X_train, X_test, y_train, y_test, features)
        results.append(r)
        save_model(m, "linear_regression", output_dir)
        save_importances(m, "linear_regression", features, output_dir)
        if not skip_robustness:
            rob["linear_regression"] = test_feature_dropout(m, X_test, y_test, features, "linear_regression")
    
    if "random_forest" in models:
        if verbose: print("  Training random_forest...")
        m = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        r = train_and_evaluate(m, "random_forest", X_train, X_test, y_train, y_test, features)
        results.append(r)
        save_model(m, "random_forest", output_dir)
        save_importances(m, "random_forest", features, output_dir)
        if not skip_robustness:
            rob["random_forest"] = test_feature_dropout(m, X_test, y_test, features, "random_forest")

    if "xgboost" in models:
        if verbose: print("  Training xgboost...")
        m = XGBRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            learning_rate=0.05, 
            subsample=0.9, 
            colsample_bytree=0.9,
            random_state=random_state, 
            verbosity=0
        )

        r = train_and_evaluate(m, "xgboost", X_train, X_test, y_train, y_test, features)
        results.append(r)
        save_model(m, "xgboost", output_dir)
        save_importances(m, "xgboost", features, output_dir)
        if not skip_robustness:
            rob["xgboost"] = test_feature_dropout(m, X_test, y_test, features, "xgboost")
    
    if len(results) == 0:
        return None
    
    if verbose: print("  Saving results...")
    save_importance_summary(results, features, output_dir)
    if len(rob) > 0:
        save_robustness_results(rob, output_dir)
    
    if not skip_plots:
        if verbose: print("  Making plots...")
        plot_feature_importance(results, features, plots_dir)
        
        dist = df.loc[X_test.index, "distance_km"].to_numpy()
        true_norm = df.loc[X_test.index, "pts_per_km"].to_numpy()
        
        for r in results:
            plot_predictions(y_test, r["y_pred"], r["model_name"], plots_dir,
                           distance_km_test=dist, y_test_original=true_norm)
        
        if len(rob) > 0:
            plot_robustness(rob, plots_dir)
    
    best = get_best_model(results)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(y),
        "n_features": len(features),
        "feature_names": features,
        "best_model": best["model_name"],
        "best_r2": round(best["r2"], 4),
        "results": results,
        "models": [],
    }
    
    for r in results:
        summary["models"].append({
            "name": r["model_name"], 
            "r2": round(r["r2"], 4), 
            "mae": round(r["mae"], 4), 
            "rmse": round(r["rmse"], 4)
        })
    
    to_save = {}
    for k in summary:
        if k != "results":
            to_save[k] = summary[k]
    
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(to_save, f, indent=2)
    
    return summary
