"""
Phase 1-2: Data Loading + Preprocessing Pipeline

Loads the raw incident dataset, engineers features that would realistically be
available at ticket-intake time, encodes categoricals, splits train/test, and
saves everything needed for Phase 3 (model training).

Deliberately EXCLUDES post-incident fields (Resolved_Time, Resolution_Time_Hours,
Status, Resolution_Type) from the feature set to avoid data leakage — none of
those exist yet when a ticket is first submitted.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

RAW_PATH = "../data/raw/it_incident_dataset.csv"
PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"

CATEGORICAL_FEATURES = ["Incident_Type", "Assigned_Department", "Location"]
TARGET = "Priority"
RANDOM_STATE = 42


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Basic cleaning: drop exact duplicates, drop rows with missing target."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows during cleaning")
    return df


def engineer_time_features(df):
    """Extract intake-time-known features from Reported_Time."""
    df = df.copy()
    reported = pd.to_datetime(df["Reported_Time"])
    df["report_hour"] = reported.dt.hour
    df["report_day_of_week"] = reported.dt.dayofweek  # 0=Monday
    df["report_month"] = reported.dt.month
    df["report_is_weekend"] = (reported.dt.dayofweek >= 5).astype(int)
    return df


def build_feature_frame(df):
    """Select only the columns that are known at ticket-intake time."""
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
    """One-hot encode categorical columns, leave engineered numeric columns as-is."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_FEATURES])

    def transform(X):
        cat_encoded = encoder.transform(X[CATEGORICAL_FEATURES])
        cat_cols = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=X.index)
        numeric_df = X.drop(columns=CATEGORICAL_FEATURES)
        return pd.concat([cat_df, numeric_df], axis=1)

    X_train_enc = transform(X_train)
    X_test_enc = transform(X_test)
    return X_train_enc, X_test_enc, encoder


def encode_target(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    return y_train_enc, y_test_enc, label_encoder


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows")

    print("Cleaning data...")
    df = clean_data(df)

    print("Building feature frame (intake-time features only)...")
    X, y = build_feature_frame(df)
    print(f"  Features: {list(X.columns)}")

    print("Splitting train/test (80/20, stratified on Priority)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Encoding features...")
    X_train_enc, X_test_enc, feature_encoder = encode_features(X_train, X_test)

    print("Encoding target...")
    y_train_enc, y_test_enc, label_encoder = encode_target(y_train, y_test)
    print(f"  Classes: {list(label_encoder.classes_)}")

    print("Saving processed data + encoders...")
    X_train_enc.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test_enc.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    pd.Series(y_train_enc, name=TARGET).to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    pd.Series(y_test_enc, name=TARGET).to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    joblib.dump(feature_encoder, f"{MODELS_DIR}/feature_encoder.pkl")
    joblib.dump(label_encoder, f"{MODELS_DIR}/label_encoder.pkl")

    # Save the exact feature column order + engineering logic reference,
    # so the FastAPI inference layer in Phase 6 stays consistent with training.
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
        "target_classes": list(label_encoder.classes_),
    }
    with open(f"{MODELS_DIR}/preprocessing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Outputs written to:")
    print(f"  {PROCESSED_DIR}/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"  {MODELS_DIR}/feature_encoder.pkl, label_encoder.pkl, preprocessing_meta.json")


if __name__ == "__main__":
    main()
