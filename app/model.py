"""
Phase 6: Model loading + inference logic.

Loads the artifacts produced in Phase 2-3 exactly once at startup and applies
the SAME feature engineering used during training, so there's no
training/serving skew. Column order is enforced via the saved
final_feature_columns list in each preprocessing_meta*.json file.
"""
import json
import logging
import os
from datetime import datetime

import joblib
import pandas as pd

logger = logging.getLogger("incident_intelligence")

# Resolve relative to this file's location, NOT the process's working
# directory -- cwd varies between local dev (run from app/) and Docker
# (WORKDIR may differ), so a hardcoded "../models" is fragile.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(_APP_DIR, "..", "models"))

# Note on the honest limitation of this model, surfaced to API consumers:
CONFIDENCE_NOTE = (
    "This model was trained on a dataset where Priority and Resolution_Time_Hours "
    "show very weak correlation with Incident_Type, Assigned_Department, Location, "
    "or reported time (see training_summary.json / training_summary_reg.json). "
    "Treat this prediction as a rough heuristic, not a reliable estimate."
)


class IncidentPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        logger.info("Loading model artifacts...")

        # Priority classification artifacts
        self.priority_model = joblib.load(f"{self.models_dir}/best_model.pkl")
        self.priority_feature_encoder = joblib.load(f"{self.models_dir}/feature_encoder.pkl")
        self.priority_label_encoder = joblib.load(f"{self.models_dir}/label_encoder.pkl")
        with open(f"{self.models_dir}/preprocessing_meta.json") as f:
            self.priority_meta = json.load(f)

        # Resolution time regression artifacts
        self.resolution_model = joblib.load(f"{self.models_dir}/best_model_reg.pkl")
        self.resolution_feature_encoder = joblib.load(f"{self.models_dir}/feature_encoder_reg.pkl")
        with open(f"{self.models_dir}/preprocessing_meta_reg.json") as f:
            self.resolution_meta = json.load(f)

        logger.info("Model artifacts loaded successfully.")

    @property
    def is_ready(self) -> bool:
        return all(
            [
                self.priority_model is not None,
                self.resolution_model is not None,
            ]
        )

    @staticmethod
    def _time_features(reported_time: datetime) -> dict:
        return {
            "report_hour": reported_time.hour,
            "report_day_of_week": reported_time.weekday(),
            "report_month": reported_time.month,
            "report_is_weekend": int(reported_time.weekday() >= 5),
        }

    def _predict_priority(self, incident_type, assigned_department, location, reported_time) -> str:
        row = {
            "Incident_Type": incident_type,
            "Assigned_Department": assigned_department,
            "Location": location,
            **self._time_features(reported_time),
        }
        X = pd.DataFrame([row])
        cat_cols = self.priority_meta["categorical_features"]
        encoded = self.priority_feature_encoder.transform(X[cat_cols])
        encoded_cols = self.priority_feature_encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
        numeric_df = X.drop(columns=cat_cols)
        full = pd.concat([encoded_df, numeric_df], axis=1)
        full = full.reindex(columns=self.priority_meta["final_feature_columns"], fill_value=0)

        pred_idx = self.priority_model.predict(full)[0]
        return self.priority_label_encoder.inverse_transform([pred_idx])[0]

    def _predict_resolution_time(self, incident_type, assigned_department, location, priority, reported_time) -> float:
        row = {
            "Incident_Type": incident_type,
            "Assigned_Department": assigned_department,
            "Location": location,
            "Priority": priority,
            **self._time_features(reported_time),
        }
        X = pd.DataFrame([row])
        cat_cols = self.resolution_meta["categorical_features"]
        encoded = self.resolution_feature_encoder.transform(X[cat_cols])
        encoded_cols = self.resolution_feature_encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
        numeric_df = X.drop(columns=cat_cols)
        full = pd.concat([encoded_df, numeric_df], axis=1)
        full = full.reindex(columns=self.resolution_meta["final_feature_columns"], fill_value=0)

        pred = self.resolution_model.predict(full)[0]
        return max(0.0, round(float(pred), 2))  # resolution time can't be negative

    def predict(self, incident_type, assigned_department, location, reported_time=None, priority=None):
        reported_time = reported_time or datetime.now()

        if priority is not None:
            priority_value = priority
            priority_source = "provided"
        else:
            priority_value = self._predict_priority(
                incident_type, assigned_department, location, reported_time
            )
            priority_source = "predicted"

        resolution_hours = self._predict_resolution_time(
            incident_type, assigned_department, location, priority_value, reported_time
        )

        return {
            "priority": priority_value,
            "priority_source": priority_source,
            "estimated_resolution_time_hours": resolution_hours,
            "model_confidence_note": CONFIDENCE_NOTE,
        }
