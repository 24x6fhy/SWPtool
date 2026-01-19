# save results

import pandas as pd
import joblib
from pathlib import Path


def save_model(model, name, where):
    where = Path(where)
    joblib.dump(model, where / f"{name}.joblib")


def save_importances(model, name, features, where):
    where = Path(where)
    
    # tree models have feature_importances_
    if hasattr(model, "feature_importances_"):
        rows = []
        for i in range(len(features)):
            rows.append({"feature": features[i], "importance": model.feature_importances_[i]})
        
        # sort by importance
        rows.sort(key=lambda x: x["importance"], reverse=True)
        
        df = pd.DataFrame(rows)
        df.to_csv(where / f"{name}_importances.csv", index=False)
        
    # linear models have coef_
    elif hasattr(model, "coef_"):
        rows = []
        for i in range(len(features)):
            rows.append({"feature": features[i], "coefficient": model.coef_[i]})
        
        # sort by absolute value
        rows.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        
        df = pd.DataFrame(rows)
        df.to_csv(where / f"{name}_coefficients.csv", index=False)


def save_importance_summary(results, features, where):
    where = Path(where)
    rows = []
    
    for res in results:
        m = res["model"]
        mname = res["model_name"]
        
        # skip linear regression
        if mname == "linear_regression":
            continue
        
        if hasattr(m, "feature_importances_"):
            for i in range(len(features)):
                rows.append({
                    "model": mname,
                    "feature": features[i],
                    "importance": m.feature_importances_[i],
                })
    
    if len(rows) == 0:
        return
    
    df = pd.DataFrame(rows)
    
    # add rank within each model
    for mname in df["model"].unique():
        mask = df["model"] == mname
        subset = df[mask].copy()
        subset = subset.sort_values("importance", ascending=False)
        for rank, idx in enumerate(subset.index, 1):
            df.loc[idx, "rank"] = rank
    
    df["rank"] = df["rank"].astype(int)
    df = df.sort_values(["model", "rank"])
    df.to_csv(where / "feature_importance_summary.csv", index=False)


def save_robustness_results(data, where):
    where = Path(where)
    
    rows = []
    for mname in data:
        for r in data[mname]:
            rows.append({
                "model": mname,
                "scenario": r["scenario"],
                "r2": r["r2"],
                "rmse": r["rmse"],
                "r2_degradation": r["r2_degradation"],
                "rmse_increase_pct": r["rmse_increase_pct"],
            })
    
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_csv(where / "robustness_feature_dropout.csv", index=False)
