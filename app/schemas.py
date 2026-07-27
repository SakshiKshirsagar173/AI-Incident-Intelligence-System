"""
Phase 6: Pydantic schemas for request validation and response shape.

Categorical enums are hardcoded from the values seen in the training data
(models/preprocessing_meta*.json). If the real dataset later grows new
category values, these enums -- and the encoders -- need retraining, not
just an edit here.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IncidentType(str, Enum):
    application_bug = "Application Bug"
    database_failure = "Database Failure"
    network_outage = "Network Outage"
    security_breach = "Security Breach"
    server_crash = "Server Crash"


class Department(str, Enum):
    database_admin = "Database Admin"
    devops = "DevOps"
    it_support = "IT Support"
    network_team = "Network Team"
    security_team = "Security Team"


class Location(str, Enum):
    data_center_a = "Data Center A"
    data_center_b = "Data Center B"
    head_office = "Head Office"
    remote_site_1 = "Remote Site 1"
    remote_site_2 = "Remote Site 2"


class Priority(str, Enum):
    critical = "Critical"
    high = "High"
    low = "Low"
    medium = "Medium"


class IncidentRequest(BaseModel):
    incident_type: IncidentType
    assigned_department: Department
    location: Location
    reported_time: Optional[datetime] = Field(
        default=None,
        description="Defaults to now (server time) if not provided.",
    )
    priority: Optional[Priority] = Field(
        default=None,
        description=(
            "If provided, used as-is for the resolution-time estimate. "
            "If omitted, priority is predicted by the classification model "
            "and that prediction is used instead."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "incident_type": "Network Outage",
                "assigned_department": "Network Team",
                "location": "Data Center A",
                "reported_time": "2026-07-27T09:00:00",
            }
        }


class IncidentResponse(BaseModel):
    priority: str
    priority_source: str  # "predicted" or "provided"
    estimated_resolution_time_hours: float
    model_confidence_note: str


class HealthResponse(BaseModel):
    status: str
    priority_model_loaded: bool
    resolution_time_model_loaded: bool
