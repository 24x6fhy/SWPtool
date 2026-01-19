# Plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error


def plot_feature_importance(results, features, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    # skip lr
    to_plot = []
    for r in results:
        if r["model_name"] != "linear_regression":
            to_plot.append(r)
    
    if len(to_plot) == 0:
        print("nothing to plot")
        return
    
    n = len(to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    
    for i in range(len(to_plot)):
        ax = axes[i]
        model = to_plot[i]["model"]
        name = to_plot[i]["model_name"]
        
        if not hasattr(model, "feature_importances_"):
            continue
        
        data = {"feature": features, "importance": model.feature_importances_}
        df = pd.DataFrame(data)
        df = df.sort_values("importance", ascending=True)
        
        ax.barh(df["feature"], df["importance"], alpha=0.7, color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance_comparison.png", dpi=150)
    plt.close()


def plot_predictions(y_test, y_pred, model_name, plots_dir, distance_km_test=None, y_test_original=None):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6)
    
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', label='perfect')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{model_name} - Raw')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    ax.text(0.05, 0.95, f'R²={r2:.3f}\nRMSE={rmse:.1f}', transform=ax.transAxes, 
            verticalalignment='top', fontsize=10)
    
    if distance_km_test is not None and y_test_original is not None:
        ax = axes[1]
        pred_norm = y_pred / distance_km_test
        true_norm = y_test_original
        
        ax.scatter(true_norm, pred_norm, alpha=0.6, color='green')
        lo = min(true_norm.min(), pred_norm.min())
        hi = max(true_norm.max(), pred_norm.max())
        ax.plot([lo, hi], [lo, hi], 'r--', label='perfect')
        ax.set_xlabel('Actual pts/km')
        ax.set_ylabel('Predicted pts/km')
        ax.set_title(f'{model_name} - Normalized')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        r2_n = r2_score(true_norm, pred_norm)
        rmse_n = mean_squared_error(true_norm, pred_norm) ** 0.5
        ax.text(0.05, 0.95, f'R²={r2_n:.3f}\nRMSE={rmse_n:.3f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_predictions.png", dpi=150)
    plt.close()


def plot_robustness(data, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    scenarios = []
    for model in data:
        for r in data[model]:
            if r["scenario"] not in scenarios:
                scenarios.append(r["scenario"])
    scenarios.sort()
    
    if len(scenarios) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    w = 0.25
    x = np.arange(len(scenarios))

    ax = axes[0]
    i = 0
    for model in data:
        vals = []
        for s in scenarios:
            # find matching scenario
            found = None
            for r in data[model]:
                if r["scenario"] == s:
                    found = r
                    break
            if found:
                vals.append(found["r2_degradation"])
            else:
                vals.append(0)
        ax.bar(x + i * w, vals, w, label=model, alpha=0.8)
        i += 1
    
    ax.set_xlabel('Missing Sensor')
    ax.set_ylabel('R² Drop')
    ax.set_title('Robustness Test')
    ax.set_xticks(x + w)
    labels = [s.replace("no_", "") for s in scenarios]
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    i = 0
    for model in data:
        vals = []
        for s in scenarios:
            found = None
            for r in data[model]:
                if r["scenario"] == s:
                    found = r
                    break
            if found:
                vals.append(found["rmse_increase_pct"])
            else:
                vals.append(0)
        ax.bar(x + i * w, vals, w, label=model, alpha=0.8)
        i += 1
    
    ax.set_xlabel('Missing Sensor')
    ax.set_ylabel('RMSE Increase %')
    ax.set_title('Robustness - RMSE')
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "robustness_comparison.png", dpi=150)
    plt.close()
