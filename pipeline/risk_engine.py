"""Spatial near-miss and time-to-collision analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    score: float
    minimum_distance: float
    ttc_seconds: float | None
    conflicting_rider_id: int | None
    reason: str

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


class RiskEngine:
    """Compare projected paths and classify risk with deterministic thresholds."""

    def __init__(self, caution_distance: float = 45.0, collision_distance: float = 18.0, ttc_warning: float = 2.5) -> None:
        if not 0 < collision_distance < caution_distance or ttc_warning <= 0:
            raise ValueError("risk thresholds are inconsistent")
        self.caution_distance = caution_distance
        self.collision_distance = collision_distance
        self.ttc_warning = ttc_warning

    @staticmethod
    def _points(path: Iterable[Iterable[float]]) -> np.ndarray:
        points = np.asarray(list(path), dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("future paths must contain [x, y] coordinates")
        return points

    def assess(self, ego_path: Iterable[Iterable[float]], neighbors: dict[int, Iterable[Iterable[float]]] | None = None, *, frame_rate: float = 30.0) -> RiskAssessment:
        ego = self._points(ego_path)
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if len(ego) < 2:
            return RiskAssessment("green", 0.0, float("inf"), None, None, "Insufficient projected path")
        best = RiskAssessment("green", 0.0, float("inf"), None, None, "Nominal path")
        for rider_id, path in (neighbors or {}).items():
            other = self._points(path)
            if len(other) == 0:
                continue
            count = min(len(ego), len(other))
            distances = np.linalg.norm(ego[:count] - other[:count], axis=1)
            index = int(np.argmin(distances))
            distance = float(distances[index])
            ttc = index / frame_rate
            score = float(np.clip(1.0 - distance / self.caution_distance, 0.0, 1.0))
            if distance <= self.collision_distance and ttc <= self.ttc_warning:
                candidate = RiskAssessment("red", max(score, 0.9), distance, ttc, int(rider_id), "Projected collision conflict")
            elif distance <= self.caution_distance or ttc <= self.ttc_warning:
                candidate = RiskAssessment("yellow", max(score, 0.45), distance, ttc, int(rider_id), "Spatial proximity or low TTC")
            else:
                candidate = RiskAssessment("green", score * 0.25, distance, ttc, int(rider_id), "Nominal path")
            if (candidate.level == "red", candidate.level == "yellow", -candidate.score) > (best.level == "red", best.level == "yellow", -best.score):
                best = candidate
        return best

    def project_path(self, origin: tuple[float, float], velocity: tuple[float, float], steps: int, frame_rate: float = 30.0) -> list[list[float]]:
        if steps < 1 or frame_rate <= 0:
            raise ValueError("steps must be positive and frame_rate must be positive")
        start = np.asarray(origin, dtype=np.float64)
        delta = np.asarray(velocity, dtype=np.float64) / frame_rate
        return (start + np.arange(1, steps + 1)[:, None] * delta).round(4).tolist()
