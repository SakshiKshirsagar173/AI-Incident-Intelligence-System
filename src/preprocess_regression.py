"""
Phase 1-2 (Regression variant): Preprocessing for Resolution_Time_Hours

Same intake-time-only philosophy as preprocess.py, but for a regression target.
Priority IS included as a feature here (unlike in preprocess.py) because in a
real workflow, priority is assigned by the agent/system at ticket creation,
before resolution time is known -- so it's legitimate signal for this target.

Still excludes Resolved_Time, Status, Resolution_Type -- those only exist after
the incident is closed.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

RAW_PATH = "../data/raw/it_incident_dataset.csv"
PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"

CATEGORICAL_FEATURES = ["Incident_Type", "Assigned_Department", "Location", "Priority"]
TARGET = "Resolution_Time_Hours"
RANDOM_STATE = 42


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def clean_data(df):
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows during cleaning")
    return df


def engineer_time_features(df):
    df = df.copy()
    reported = pd.to_datetime(df["Reported_Time"])
    df["report_hour"] = reported.dt.hour
    df["report_day_of_week"] = reported.dt.dayofweek
    df["report_month"] = reported.dt.month
    df["report_is_weekend"] = (reported.dt.dayofweek >= 5).astype(int)
    return df


def build_feature_frame(df):
    df = engineer_time_features(df)
    feature_cols = CATEGORICAL_FEATURES + [
        "report_hour",
        "report_day_of_week",
        "report_month",
        "report_is_weekend",
    ]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


def encode_features(X_train, X_test):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_FEATURES])

    def transform(X):
        cat_encoded = encoder.transform(X[CATEGORICAL_FEATURES])
        cat_cols = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=X.index)
        numeric_df = X.drop(columns=CATEGORICAL_FEATURES)
        return pd.concat([cat_df, numeric_df], axis=1)

    return transform(X_train), transform(X_test), encoder


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Building feature frame (intake-time features + Priority)...")
    X, y = build_feature_frame(df)
    print(f"  Features: {list(X.columns)}")

    print("Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Encoding features...")
    X_train_enc, X_test_enc, feature_encoder = encode_features(X_train, X_test)

    print("Saving processed data + encoder...")
    X_train_enc.to_csv(f"{PROCESSED_DIR}/X_train_reg.csv", index=False)
    X_test_enc.to_csv(f"{PROCESSED_DIR}/X_test_reg.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train_reg.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test_reg.csv", index=False)

    joblib.dump(feature_encoder, f"{MODELS_DIR}/feature_encoder_reg.pkl")

    meta = {
        "categorical_features": CATEGORICAL_FEATURES,
        "engineered_numeric_features": [
            "report_hour",
            "report_day_of_week",
            "report_month",
            "report_is_weekend",
        ],
        "target": TARGET,
        "final_feature_columns": list(X_train_enc.columns),
    }
    with open(f"{MODELS_DIR}/preprocessing_meta_reg.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Outputs written to:")
    print(f"  {PROCESSED_DIR}/X_train_reg.csv, X_test_reg.csv, y_train_reg.csv, y_test_reg.csv")
    print(f"  {MODELS_DIR}/feature_encoder_reg.pkl, preprocessing_meta_reg.json")


if __name__ == "__main__":
    main()
