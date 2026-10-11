"""Production-oriented FastAPI service for GARUDA trajectory inference."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models.intent_forecaster import IntentForecaster
from pipeline.feature_engine import FeatureEngine, RiderFrame
from pipeline.risk_engine import RiskEngine


class TrajectoryPoint(BaseModel):
    timestamp: float = Field(ge=0)
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0
    head_yaw: float = 0.0
    torso_lean: float = 0.0
    arm_signal: float = 0.0


class TrajectoryRequest(BaseModel):
    rider_id: int = Field(default=0, ge=0)
    history: list[TrajectoryPoint] = Field(default_factory=list, min_length=1, max_length=90)
    neighbors: dict[int, list[list[float]]] = Field(default_factory=dict)
    engine: str = Field(default="baseline", pattern="^(baseline|xgboost|lstm|gru|transformer)$")
    horizon_seconds: float = Field(default=2.0, description="Supported values: 1.0, 2.0, 3.0")
    video_slice_base64: str | None = None


class TelemetryEvent(BaseModel):
    event: str
    rider_id: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class GarudaService:
    def __init__(self) -> None:
        self.features = FeatureEngine()
        self.risk = RiskEngine()
        self._logs: deque[dict[str, Any]] = deque(maxlen=5000)
        self._lock = Lock()

    def predict(self, request: TrajectoryRequest) -> dict[str, Any]:
        if request.video_slice_base64 and not request.history:
            raise HTTPException(status_code=501, detail="Video slices require a configured detector; send trajectory history for the local baseline")
        try:
            self.features.remove(request.rider_id)
            for point in request.history:
                self.features.update(request.rider_id, RiderFrame(**point.model_dump()))
            window = self.features.window(request.rider_id)
            latest = self.features.extract(request.rider_id)
            sequence_engine = FeatureEngine()
            sequence_features = []
            for frame in window:
                sequence_engine.update(request.rider_id, frame)
                current = sequence_engine.extract(request.rider_id)
                sequence_features.append(current.vector())
            history = np.asarray(sequence_features, dtype=np.float32)
            forecaster = IntentForecaster(request.horizon_seconds)
            forecast = forecaster.forecast(history, request.rider_id, origin=window[-1].bottom_center, engine=request.engine)
            risk = self.risk.assess(forecast.future_path, request.neighbors)
            return {"forecast": forecast.to_dict(), "features": latest.to_dict(), "risk": risk.to_dict(), "timestamp": datetime.now(timezone.utc).isoformat()}
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def log(self, event: TelemetryEvent) -> dict[str, Any]:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event.model_dump()}
        with self._lock:
            self._logs.append(entry)
        return entry


service = GarudaService()
app = FastAPI(title="PROJECT GARUDA Intent Safety API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "garuda-intent-safety"}


@app.post("/predict/trajectory")
def predict_trajectory(request: TrajectoryRequest) -> dict[str, Any]:
    return service.predict(request)


@app.post("/telemetry/log")
def log_telemetry(event: TelemetryEvent) -> dict[str, Any]:
    return {"status": "accepted", "event": service.log(event)}
