"""Optional YOLO pose detection with ByteTrack-compatible rider tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RiderDetection:
    rider_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    keypoints: np.ndarray | None = field(default=None, repr=False)
    head_yaw: float = 0.0
    torso_lean: float = 0.0
    arm_signal: float = 0.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class RiderDetectorTracker:
    """Run YOLO when installed and retain a lightweight fallback tracker.

    ``ultralytics`` and ``supervision`` are imported lazily so API and benchmark
    workflows remain usable on machines without model weights or GPU support.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt", classes: list[int] | None = None, confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.classes = classes if classes is not None else [0]
        self.confidence = confidence
        self._model: Any | None = None
        self._tracker: Any | None = None
        self._lock = RLock()
        self._next_id = 1
        self._last_centers: dict[int, np.ndarray] = {}

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("ultralytics is required for video detection; install requirements.txt") from error
            self._model = YOLO(self.model_path)
            try:
                from supervision import ByteTrack
                self._tracker = ByteTrack()
            except ImportError:
                self._tracker = None

    @staticmethod
    def _pose_cues(keypoints: np.ndarray | None) -> tuple[float, float, float]:
        if keypoints is None or len(keypoints) < 7:
            return 0.0, 0.0, 0.0
        points = np.asarray(keypoints, dtype=np.float32)
        # COCO pose indices: nose, shoulders, elbows, wrists, hips.
        nose, left_shoulder, right_shoulder = points[0], points[5], points[6]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        head_yaw = float(np.clip((nose[0] - shoulder_mid[0]) / max(np.linalg.norm(left_shoulder - right_shoulder), 1.0) * 90.0, -90.0, 90.0))
        torso_lean = float(np.degrees(np.arctan2(shoulder_mid[0] - ((points[11, 0] + points[12, 0]) / 2.0 if len(points) > 12 else shoulder_mid[0]), max(abs(shoulder_mid[1] - nose[1]), 1.0))))
        arm_signal = float(np.clip(np.linalg.norm(points[9] - points[7]) if len(points) > 9 else 0.0, 0.0, 1.0))
        return head_yaw, torso_lean, arm_signal

    def _fallback_ids(self, boxes: list[tuple[float, float, float, float]]) -> list[int]:
        assigned: list[int] = []
        next_centers: dict[int, np.ndarray] = {}
        for box in boxes:
            center = np.asarray(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0), dtype=np.float32)
            choices = [(float(np.linalg.norm(center - old)), rider_id) for rider_id, old in self._last_centers.items() if rider_id not in assigned]
            rider_id = min(choices)[1] if choices and min(choices)[0] < 80.0 else self._next_id
            if rider_id == self._next_id:
                self._next_id += 1
            assigned.append(rider_id)
            next_centers[rider_id] = center
        self._last_centers = next_centers
        return assigned

    def detect(self, frame: np.ndarray) -> list[RiderDetection]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a color numpy array")
        with self._lock:
            if self._model is None:
                self.load()
            results = self._model.predict(frame, classes=self.classes, conf=self.confidence, verbose=False)
            result = results[0]
            boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
            scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty(0)
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.empty(0, dtype=int)
            poses = result.keypoints.xy.detach().cpu().numpy() if getattr(result, "keypoints", None) is not None else []
            ids = result.boxes.id.detach().cpu().numpy().astype(int).tolist() if getattr(result.boxes, "id", None) is not None else None
            if self._tracker is not None and len(boxes):
                from supervision import Detections
                tracked = self._tracker.update_with_detections(Detections(xyxy=boxes, confidence=scores, class_id=class_ids))
                boxes = tracked.xyxy
                scores = tracked.confidence if tracked.confidence is not None else np.ones(len(boxes))
                class_ids = tracked.class_id if tracked.class_id is not None else np.zeros(len(boxes), dtype=int)
                ids = tracked.tracker_id.tolist() if tracked.tracker_id is not None else None
            if ids is None:
                ids = self._fallback_ids([tuple(map(float, box)) for box in boxes])
            detections: list[RiderDetection] = []
            for index, box in enumerate(boxes):
                keypoints = np.asarray(poses[index]) if index < len(poses) else None
                head_yaw, torso_lean, arm_signal = self._pose_cues(keypoints)
                detections.append(RiderDetection(int(ids[index]), tuple(map(float, box)), float(scores[index]), int(class_ids[index]), keypoints, head_yaw, torso_lean, arm_signal))
            return detections
