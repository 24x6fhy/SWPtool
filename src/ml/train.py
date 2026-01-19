# training models

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate(model, name, X_train, X_test, y_train, y_test, features):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # find most important feature
    mvp = None
    if hasattr(model, "feature_importances_"):
        idx = 0
        best = model.feature_importances_[0]
        for i in range(len(model.feature_importances_)):
            if model.feature_importances_[i] > best:
                best = model.feature_importances_[i]
                idx = i
        mvp = features[idx]
    elif hasattr(model, "coef_"):
        # linear model
        idx = 0
        best = abs(model.coef_[0])
        for i in range(len(model.coef_)):
            if abs(model.coef_[i]) > best:
                best = abs(model.coef_[i])
                idx = i
        mvp = features[idx]

    return {
        "model_name": name,
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "r2": r2_score(y_test, pred),
        "top_feature": mvp,
        "model": model,
        "y_pred": pred,
        "y_test": y_test.to_numpy(),
    }


def get_best_model(results):
    # just loop through and find highest r2
    best = results[0]
    for r in results:
        if r["r2"] > best["r2"]:
            best = r
    return best
