"""
Phase 1: Exploratory Data Analysis
Run this first to understand the dataset before building the preprocessing pipeline.
"""
import pandas as pd

DATA_PATH = "../data/raw/it_incident_dataset.csv"


def run_eda():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("DTYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(df.duplicated().sum())

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION (Priority)")
    print("=" * 60)
    print(df["Priority"].value_counts())
    print(df["Priority"].value_counts(normalize=True).round(3))

    print("\n" + "=" * 60)
    print("PRIORITY BY INCIDENT TYPE (cross-tab, %)")
    print("=" * 60)
    ct = pd.crosstab(df["Incident_Type"], df["Priority"], normalize="index").round(2)
    print(ct)

    print("\n" + "=" * 60)
    print("PRIORITY BY DEPARTMENT (cross-tab, %)")
    print("=" * 60)
    ct2 = pd.crosstab(df["Assigned_Department"], df["Priority"], normalize="index").round(2)
    print(ct2)

    print("\n" + "=" * 60)
    print("REPORTED_TIME RANGE")
    print("=" * 60)
    rt = pd.to_datetime(df["Reported_Time"])
    print(f"From {rt.min()} to {rt.max()}")

    print("\nNOTE: Resolved_Time, Resolution_Time_Hours, Status, and Resolution_Type")
    print("are all POST-INCIDENT fields. They are excluded from the feature set for")
    print("Priority prediction to avoid data leakage (they wouldn't exist yet at the")
    print("moment a real ticket is submitted).")


if __name__ == "__main__":
    run_eda()
