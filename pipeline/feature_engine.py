"""Thread-safe trajectory buffering and kinematic feature extraction."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot
from threading import RLock
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RiderFrame:
    """One tracked rider observation in image or ground-plane coordinates."""

    timestamp: float
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0
    head_yaw: float = 0.0
    torso_lean: float = 0.0
    arm_signal: float = 0.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class TrajectoryFeatures:
    rider_id: int
    timestamp: float
    speed: float
    acceleration: float
    heading: float
    heading_rate: float
    head_yaw_rate: float
    torso_lean: float
    arm_signal: float
    sample_count: int

    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.speed, self.acceleration, self.heading, self.heading_rate,
             self.head_yaw_rate, self.torso_lean, self.arm_signal], dtype=np.float32
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class FeatureEngine:
    """Maintain bounded per-rider windows and derive stable numeric features."""

    def __init__(self, maxlen: int = 90, fps: float = 30.0) -> None:
        if maxlen < 2 or fps <= 0:
            raise ValueError("maxlen must be >= 2 and fps must be positive")
        self.maxlen = maxlen
        self.fps = fps
        self._buffers: dict[int, deque[RiderFrame]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._lock = RLock()

    def update(self, rider_id: int, frame: RiderFrame) -> TrajectoryFeatures:
        with self._lock:
            self._buffers[int(rider_id)].append(frame)
            return self.extract(rider_id)

    def extend(self, rider_id: int, frames: Iterable[RiderFrame]) -> TrajectoryFeatures:
        result: TrajectoryFeatures | None = None
        for frame in frames:
            result = self.update(rider_id, frame)
        if result is None:
            raise ValueError("frames must contain at least one observation")
        return result

    def extract(self, rider_id: int) -> TrajectoryFeatures:
        with self._lock:
            samples = list(self._buffers.get(int(rider_id), ()))
        if not samples:
            raise KeyError(f"no trajectory samples for rider {rider_id}")
        if len(samples) < 2:
            return TrajectoryFeatures(int(rider_id), samples[-1].timestamp, 0.0, 0.0, 0.0, 0.0, 0.0, samples[-1].torso_lean, samples[-1].arm_signal, 1)

        deltas = np.diff(np.asarray([sample.bottom_center for sample in samples], dtype=np.float64), axis=0)
        times = np.diff(np.asarray([sample.timestamp for sample in samples], dtype=np.float64))
        times = np.where(times > 1e-6, times, 1.0 / self.fps)
        distances = np.linalg.norm(deltas, axis=1)
        speeds = distances / times
        headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        heading_rates = np.diff(headings) / times[1:] if len(headings) > 1 else np.zeros(1)
        accelerations = np.diff(speeds) / times[1:] if len(speeds) > 1 else np.zeros(1)
        yaw_rates = np.diff(np.asarray([sample.head_yaw for sample in samples])) / times
        latest_heading = degrees(float(headings[-1]))
        return TrajectoryFeatures(
            rider_id=int(rider_id), timestamp=float(samples[-1].timestamp),
            speed=float(speeds[-1]), acceleration=float(accelerations[-1]),
            heading=latest_heading, heading_rate=degrees(float(heading_rates[-1])) if len(heading_rates) else 0.0,
            head_yaw_rate=float(yaw_rates[-1]) if len(yaw_rates) else 0.0,
            torso_lean=float(samples[-1].torso_lean), arm_signal=float(samples[-1].arm_signal), sample_count=len(samples),
        )

    def window(self, rider_id: int) -> list[RiderFrame]:
        with self._lock:
            return list(self._buffers.get(int(rider_id), ()))

    def remove(self, rider_id: int) -> None:
        with self._lock:
            self._buffers.pop(int(rider_id), None)

    def snapshot(self) -> dict[int, list[dict[str, float]]]:
        with self._lock:
            return {rider_id: [asdict(frame) for frame in frames] for rider_id, frames in self._buffers.items()}


def features_from_points(points: list[tuple[float, float]], fps: float = 30.0) -> np.ndarray:
    """Convenience helper for benchmarks using only point trajectories."""
    engine = FeatureEngine(fps=fps)
    for index, (x, y) in enumerate(points):
        engine.update(0, RiderFrame(index / fps, float(x), float(y)))
    return engine.extract(0).vector()
