"""
Phase 3 (Regression variant): Train & compare models for Resolution_Time_Hours

Trains Linear Regression, Random Forest Regressor, and Gradient Boosting
Regressor, evaluates with MAE, RMSE, and R^2, and saves the best by R^2.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"
RANDOM_STATE = 42


def load_processed():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_reg.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test_reg.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train_reg.csv")["Resolution_Time_Hours"]
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test_reg.csv")["Resolution_Time_Hours"]
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed()

    print("Baseline (predict mean) for reference:")
    mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, mean_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
    print(f"  MAE: {baseline_mae:.3f} | RMSE: {baseline_rmse:.3f} | R2: 0.000\n")

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        print(f"  MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | R2: {metrics['r2']:.3f}\n")

    print("=" * 60)
    print("FULL COMPARISON")
    print("=" * 60)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(3))

    best_name = comparison_df["r2"].idxmax()
    print(f"\nBest model by R2: {best_name}")

    best_model = models[best_name]
    joblib.dump(best_model, f"{MODELS_DIR}/best_model_reg.pkl")
    print(f"Saved best model to {MODELS_DIR}/best_model_reg.pkl")

    summary = {
        "baseline_mean_predictor": {"mae": round(baseline_mae, 3), "rmse": round(baseline_rmse, 3), "r2": 0.0},
        "best_model": best_name,
        "comparison": {name: {k: round(v, 3) for k, v in m.items()} for name, m in results.items()},
    }
    with open(f"{MODELS_DIR}/training_summary_reg.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {MODELS_DIR}/training_summary_reg.json")


if __name__ == "__main__":
    main()
