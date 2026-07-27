"""
Phase 3: Model Training & Comparison

Trains Logistic Regression, Linear SVM, and Random Forest on the processed
Phase 1-2 data, evaluates each on the held-out test set, and picks the best
model based on macro F1 (fair across the 4 Priority classes, which are
close to balanced but not identical).
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"
RANDOM_STATE = 42


def load_processed():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv")["Priority"]
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv")["Priority"]
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def evaluate(model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "report": classification_report(
            y_test, preds, target_names=label_encoder.classes_, zero_division=0
        ),
    }


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed()
    label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")

    print("Baseline (majority class) accuracy for reference:")
    majority_class = y_train.value_counts().idxmax()
    baseline_acc = (y_test == majority_class).mean()
    print(f"  {baseline_acc:.3f}\n")

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, label_encoder)
        results[name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f} | "
              f"Precision: {metrics['precision_macro']:.3f} | "
              f"Recall: {metrics['recall_macro']:.3f} | "
              f"F1: {metrics['f1_macro']:.3f}\n")

    print("=" * 60)
    print("FULL COMPARISON")
    print("=" * 60)
    comparison_df = pd.DataFrame(
        {name: {k: v for k, v in m.items() if k != "report"} for name, m in results.items()}
    ).T
    print(comparison_df.round(3))

    best_name = comparison_df["f1_macro"].idxmax()
    print(f"\nBest model by macro F1: {best_name}")
    print("\nClassification report for best model:")
    print(results[best_name]["report"])

    best_model = models[best_name]
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    print(f"Saved best model to {MODELS_DIR}/best_model.pkl")

    summary = {
        "baseline_majority_class_accuracy": round(baseline_acc, 3),
        "best_model": best_name,
        "comparison": {
            name: {k: round(v, 3) for k, v in m.items() if k != "report"}
            for name, m in results.items()
        },
    }
    with open(f"{MODELS_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {MODELS_DIR}/training_summary.json")


if __name__ == "__main__":
    main()
