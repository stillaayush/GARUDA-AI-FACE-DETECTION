"""Multi-engine rider intent forecasting with explainability hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import cos, radians, sin
from threading import RLock
from typing import Iterable

import numpy as np


class Intent(IntEnum):
    STRAIGHT = 0
    LEFT_TURN = 1
    RIGHT_TURN = 2
    STOP_BRAKE = 3
    LANE_CHANGE = 4


INTENT_NAMES = {item.value: item.name.replace("_", " ").title() for item in Intent}
FEATURE_NAMES = ("speed", "acceleration", "heading", "heading_rate", "head_yaw_rate", "torso_lean", "arm_signal")


@dataclass(frozen=True)
class ForecastResult:
    rider_id: int
    intent: str
    intent_id: int
    probabilities: dict[str, float]
    future_path: list[list[float]]
    horizon_seconds: float
    engine: str
    attributions: list[dict[str, float | str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "rider_id": self.rider_id, "intent": self.intent, "intent_id": self.intent_id,
            "probabilities": self.probabilities, "future_path": self.future_path,
            "horizon_seconds": self.horizon_seconds, "engine": self.engine, "attributions": self.attributions,
        }


class IntentForecaster:
    """Baseline-first forecaster with optional sequence model and SHAP integration."""

    def __init__(self, horizon_seconds: float = 2.0, fps: float = 30.0) -> None:
        self.set_horizon(horizon_seconds)
        self.fps = fps
        self._baseline = None
        self._sequence = None
        self._lock = RLock()

    def set_horizon(self, horizon_seconds: float) -> None:
        if horizon_seconds not in (1.0, 2.0, 3.0):
            raise ValueError("horizon_seconds must be 1.0, 2.0, or 3.0")
        self.horizon_seconds = float(horizon_seconds)

    @staticmethod
    def _statistical_vector(history: np.ndarray) -> np.ndarray:
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.ndim != 2 or history.shape[1] < 7:
            raise ValueError("history must have shape (timesteps, 7) or wider")
        recent = history[-min(30, len(history)):,:7]
        return np.asarray(np.concatenate((recent.mean(axis=0), recent.std(axis=0), recent[-1])), dtype=np.float32).reshape(1, -1)

    @staticmethod
    def _heuristic_probabilities(history: np.ndarray) -> np.ndarray:
        values = history[-min(30, len(history)):,:7]
        mean = values.mean(axis=0)
        score = np.zeros(5, dtype=np.float64)
        score[0] = 1.0 + max(0.0, 1.0 - abs(mean[3]) / 20.0)
        score[1] = max(0.0, -mean[3]) + max(0.0, -mean[4] / 15.0)
        score[2] = max(0.0, mean[3]) + max(0.0, mean[4] / 15.0)
        score[3] = max(0.0, -mean[1]) + max(0.0, -mean[0] / 20.0)
        score[4] = abs(mean[5]) / 15.0 + mean[6]
        score = np.exp(score - np.max(score))
        return score / max(score.sum(), 1e-9)

    def fit_baseline(self, histories: Iterable[np.ndarray], labels: Iterable[int]) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as error:
            raise RuntimeError("scikit-learn is required to train the baseline") from error
        x = np.vstack([self._statistical_vector(np.asarray(history, dtype=np.float32)) for history in histories])
        y = np.asarray(list(labels), dtype=np.int64)
        if len(x) != len(y) or len(y) < 2:
            raise ValueError("histories and labels must contain at least two aligned samples")
        with self._lock:
            self._baseline = RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=42, n_jobs=-1)
            self._baseline.fit(x, y)

    def load_baseline(self, path: str) -> None:
        import joblib
        with self._lock:
            self._baseline = joblib.load(path)

    def _probabilities(self, history: np.ndarray, engine: str) -> np.ndarray:
        engine = engine.lower()
        if engine in {"xgboost", "baseline", "random_forest"} and self._baseline is not None:
            probabilities = self._baseline.predict_proba(self._statistical_vector(history))[0]
            result = np.zeros(len(Intent), dtype=np.float64)
            result[self._baseline.classes_.astype(int)] = probabilities
            return result
        if engine in {"lstm", "gru", "transformer"} and self._sequence is not None:
            import torch
            with torch.no_grad():
                logits = self._sequence(torch.tensor(history[-90:,:7], dtype=torch.float32).unsqueeze(0))
                return torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return self._heuristic_probabilities(history)

    def _attributions(self, history: np.ndarray, engine: str, probabilities: np.ndarray) -> list[dict[str, float | str]]:
        if engine.lower() in {"baseline", "xgboost", "random_forest"} and self._baseline is not None:
            try:
                import shap
                values = shap.TreeExplainer(self._baseline).shap_values(self._statistical_vector(history))
                matrix = np.asarray(values)
                if matrix.ndim == 3:
                    matrix = matrix[int(np.argmax(probabilities)), 0]
                elif matrix.ndim == 2:
                    matrix = matrix[0]
                names = list(FEATURE_NAMES) + [f"{name}_std" for name in FEATURE_NAMES] + [f"{name}_latest" for name in FEATURE_NAMES]
                indices = np.argsort(np.abs(matrix))[::-1][:3]
                return [{"feature": names[index], "impact": round(float(matrix[index]), 4)} for index in indices]
            except (ImportError, ValueError, TypeError, AttributeError):
                pass
        values = history[-min(30, len(history)):,:7].mean(axis=0)
        indices = np.argsort(np.abs(values))[::-1][:3]
        return [{"feature": FEATURE_NAMES[index], "impact": round(float(values[index]), 4)} for index in indices]

    def forecast(self, history: np.ndarray, rider_id: int = 0, *, origin: tuple[float, float] = (0.0, 0.0), engine: str = "baseline") -> ForecastResult:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] < 7 or len(history) == 0:
            raise ValueError("history must be a non-empty array shaped (timesteps, 7)")
        with self._lock:
            probabilities = self._probabilities(history, engine)
        intent_id = int(np.argmax(probabilities))
        latest = history[-1]
        steps = max(1, round(self.horizon_seconds * self.fps))
        speed = max(0.0, float(latest[0]))
        heading = radians(float(latest[2]))
        turn_bias = float(latest[3]) * 0.01
        path = []
        for step in range(1, steps + 1):
            elapsed = step / self.fps
            angle = heading + turn_bias * elapsed
            path.append([round(origin[0] + speed * cos(angle) * elapsed, 4), round(origin[1] + speed * sin(angle) * elapsed, 4)])
        return ForecastResult(
            rider_id=int(rider_id), intent=INTENT_NAMES[intent_id], intent_id=intent_id,
            probabilities={INTENT_NAMES[index]: round(float(value), 6) for index, value in enumerate(probabilities)},
            future_path=path, horizon_seconds=self.horizon_seconds, engine=engine,
            attributions=self._attributions(history, engine, probabilities),
        )
