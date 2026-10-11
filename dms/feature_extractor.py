from dataclasses import dataclass
from collections import deque
from pathlib import Path
import cv2
import numpy as np


@dataclass
class DMSFeatures:
    landmarks: np.ndarray
    face_bbox: tuple[int, int, int, int] | None
    hand_bbox: tuple[int, int, int, int] | None
    ear: float
    mar: float
    hand_position: str
    neck_angle: float
    drowsiness_score: float
    yawn_detected: bool
    attention_status: str
    rider_status: str
    is_drowsy: bool
    is_yawning: bool
    hands_detected: int
    hands_on_bars: bool
    head_yaw: float
    head_pitch: float
    head_roll: float
    annotated_frame: np.ndarray
    has_glasses: bool = False
    has_earphones: bool = False
    phone_call_status: str = "NONE"
    is_distracted: bool = False
    gaze_zone: str = "LOOKING_AHEAD"

    @property
    def neck_yaw(self) -> float:
        return self.head_yaw

    def model_vector(self) -> np.ndarray:
        return np.array([self.ear, self.mar, float(self.hands_on_bars), self.head_yaw / 90.0], dtype=np.float32)


class DMSFeatureExtractor:
    """Fail-safe local DMS extractor with no MediaPipe or external API dependency."""

    def __init__(self, ear_threshold: float = 0.18, mar_threshold: float = 0.45) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.closed_frames = 0
        self.mar_history: deque[float] = deque(maxlen=8)
        self.was_yawning = False
        self._last_hand_bbox: tuple[int, int, int, int] | None = None
        self.face_cascade = self._create_cascade()

    @staticmethod
    def _create_cascade() -> object | None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        try:
            cascade = cv2.CascadeClassifier(str(cascade_path))
            return None if cascade.empty() else cascade
        except (cv2.error, OSError):
            return None

    @staticmethod
    def _fallback_landmarks(bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bbox
        points = np.zeros((68, 2), dtype=np.float32)
        points[:, 0] = x + width * 0.5
        points[:, 1] = y + height * 0.5
        jaw_xs = np.linspace(x + width * 0.12, x + width * 0.88, 17)
        jaw_ys = np.full(17, y + height * 0.8)
        points[0:17] = np.column_stack((jaw_xs, jaw_ys))
        for offset, center_x in ((36, x + width * 0.34), (42, x + width * 0.66)):
            eye_width, eye_height = width * 0.22, height * 0.08
            points[offset:offset + 6] = np.array([
                [center_x - eye_width / 2, y + height * 0.4],
                [center_x - eye_width * 0.25, y + height * 0.4 - eye_height],
                [center_x + eye_width * 0.25, y + height * 0.4 - eye_height],
                [center_x + eye_width / 2, y + height * 0.4],
                [center_x + eye_width * 0.25, y + height * 0.4 + eye_height],
                [center_x - eye_width * 0.25, y + height * 0.4 + eye_height],
            ], dtype=np.float32)
        mouth_width = width * 0.32
        points[48:68] = np.array([
            [x + width * 0.5 - mouth_width / 2 + index * mouth_width / 19, y + height * 0.7 + (height * 0.04 if index in (2, 6) else 0)]
            for index in range(20)
        ], dtype=np.float32)
        return points

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second))

    @staticmethod
    def _contour_in_region(gray: np.ndarray, region: tuple[int, int, int, int], min_area: float) -> np.ndarray | None:
        x, y, width, height = region
        roi = gray[y:y + height, x:x + width]
        if roi.size == 0:
            return None
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        if not candidates:
            return None
        contour = max(candidates, key=cv2.contourArea)
        contour = contour.reshape(-1, 2).astype(np.float32)
        contour[:, 0] += x
        contour[:, 1] += y
        return contour

    @classmethod
    def _landmarks_from_contours(cls, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bbox
        points = cls._fallback_landmarks(bbox)
        eye_regions = (
            (x + int(width * 0.12), y + int(height * 0.28), max(8, int(width * 0.32)), max(8, int(height * 0.23))),
            (x + int(width * 0.56), y + int(height * 0.28), max(8, int(width * 0.32)), max(8, int(height * 0.23))),
        )
        for offset, region in zip((36, 42), eye_regions):
            contour = cls._contour_in_region(gray, region, max(2.0, width * height * 0.001))
            if contour is None:
                continue
            left = contour[np.argmin(contour[:, 0])]
            right = contour[np.argmax(contour[:, 0])]
            top = contour[np.argmin(contour[:, 1])]
            bottom = contour[np.argmax(contour[:, 1])]
            center_x = (left[0] + right[0]) / 2
            points[offset:offset + 6] = np.array([
                left, top, [center_x, top[1]], right, bottom, [center_x, bottom[1]],
            ], dtype=np.float32)
        mouth_region = (x + int(width * 0.2), y + int(height * 0.57), int(width * 0.6), int(height * 0.3))
        mouth = cls._contour_in_region(gray, mouth_region, max(3.0, width * height * 0.002))
        if mouth is not None:
            left = mouth[np.argmin(mouth[:, 0])]
            right = mouth[np.argmax(mouth[:, 0])]
            top = mouth[np.argmin(mouth[:, 1])]
            bottom = mouth[np.argmax(mouth[:, 1])]
            points[48:68] = np.array([
                [left[0] + (right[0] - left[0]) * index / 19, top[1] if index < 10 else bottom[1]]
                for index in range(20)
            ], dtype=np.float32)
        return points

    @classmethod
    def _aperture_ratio(cls, gray: np.ndarray, region: tuple[int, int, int, int], min_width_ratio: float) -> float:
        roi = cls._safe_region(gray, region)
        if roi.size == 0 or min(roi.shape[:2]) < 4:
            return 0.0
        normalized = cv2.GaussianBlur(roi, (3, 3), 0)
        _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_width = roi.shape[1] * min_width_ratio
        candidates = []
        for contour in contours:
            contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)
            if contour_width >= min_width and contour_height >= 2:
                candidates.append((cv2.contourArea(contour), contour_width, contour_height))
        if not candidates:
            return 0.0
        _, aperture_width, aperture_height = max(candidates, key=lambda item: item[0])
        return float(np.clip(aperture_height / max(aperture_width, 1), 0.0, 1.0))

    def _face_measurements(self, gray: np.ndarray, bbox: tuple[int, int, int, int], landmarks: np.ndarray, frame_width: int, frame_height: int) -> tuple[float, float, float, float, float]:
        x, y, width, height = bbox
        left_eye_region = (x + int(width * 0.1), y + int(height * 0.2), int(width * 0.36), int(height * 0.3))
        right_eye_region = (x + int(width * 0.54), y + int(height * 0.2), int(width * 0.36), int(height * 0.3))
        left_ear = self._aperture_ratio(gray, left_eye_region, 0.2)
        right_ear = self._aperture_ratio(gray, right_eye_region, 0.2)
        ear = float(np.clip((left_ear + right_ear) / 2 * 1.8, 0.0, 0.6))
        mouth_region = (x + int(width * 0.16), y + int(height * 0.6), int(width * 0.68), int(height * 0.32))
        mar = self._aperture_ratio(gray, mouth_region, 0.18)
        nose = landmarks[30].copy()
        nose_contour = self._contour_in_region(gray, (x + int(width * 0.32), y + int(height * 0.34), int(width * 0.36), int(height * 0.32)), max(3.0, width * height * 0.001))
        if nose_contour is not None:
            nose = np.mean(nose_contour, axis=0)
        face_center_x = x + width * 0.5
        yaw = float(np.clip((nose[0] - face_center_x) / max(width, 1) * 90.0, -90.0, 90.0))
        pitch = float(np.clip((nose[1] - (y + height * 0.5)) / max(height, 1) * 180.0, -90.0, 90.0))
        left_eye, right_eye = landmarks[36:42], landmarks[42:48]
        roll = float(np.degrees(np.arctan2(right_eye[0, 1] - left_eye[0, 1], right_eye[0, 0] - left_eye[0, 0])))
        return ear, mar, yaw, pitch, float(np.clip(roll, -90.0, 90.0))

    @staticmethod
    def _safe_region(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = region
        frame_height, frame_width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_width, x + width), min(frame_height, y + height)
        return frame[y1:y2, x1:x2]

    @classmethod
    def _edge_features(cls, gray: np.ndarray, region: tuple[int, int, int, int]) -> tuple[float, float, int]:
        roi = cls._safe_region(gray, region)
        if roi.size == 0 or min(roi.shape[:2]) < 3:
            return 0.0, 0.0, 0
        edges = cv2.Canny(roi, 60, 140)
        gradients = cv2.Laplacian(roi, cv2.CV_32F)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return float(np.count_nonzero(edges) / edges.size), float(np.var(gradients)), len(contours)

    @classmethod
    def _detect_glasses(cls, gray: np.ndarray, bbox: tuple[int, int, int, int], landmarks: np.ndarray) -> bool:
        x, y, width, height = bbox
        eye_points = landmarks[36:48]
        eye_left, eye_top = np.min(eye_points, axis=0)
        eye_right, eye_bottom = np.max(eye_points, axis=0)
        region = (
            int(eye_left - width * 0.08),
            int(eye_top - height * 0.12),
            int((eye_right - eye_left) + width * 0.16),
            int((eye_bottom - eye_top) + height * 0.24),
        )
        density, variance, contours = cls._edge_features(gray, region)
        bridge = (int(x + width * 0.38), int(y + height * 0.29), int(width * 0.24), int(height * 0.2))
        bridge_density, bridge_variance, _ = cls._edge_features(gray, bridge)
        return bool((density > 0.12 and variance > 90.0 and contours >= 2) or (bridge_density > 0.16 and bridge_variance > 130.0))

    @classmethod
    def _detect_earphones(cls, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        x, y, width, height = bbox
        regions = (
            (int(x - width * 0.2), int(y + height * 0.28), int(width * 0.25), int(height * 0.42)),
            (int(x + width * 0.95), int(y + height * 0.28), int(width * 0.25), int(height * 0.42)),
        )
        signals = [cls._edge_features(gray, region) for region in regions]
        dark_clusters = 0
        for region in regions:
            roi = cls._safe_region(gray, region)
            if roi.size:
                _, mask = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                dark_clusters += sum(1 for contour in contours if 8.0 <= cv2.contourArea(contour) <= roi.size * 0.18)
        return bool(any(density > 0.1 and variance > 80.0 and contours >= 2 for density, variance, contours in signals) or dark_clusters >= 2)

    def _classify_phone_call(self, frame: np.ndarray, bbox: tuple[int, int, int, int] | None, landmarks: np.ndarray, mar: float, has_earphones: bool, is_yawning: bool) -> str:
        self.mar_history.append(mar)
        if bbox is None or len(landmarks) < 68:
            return "NONE"
        x, y, width, height = bbox
        if self._last_hand_bbox is not None:
            hand_x, hand_y, hand_width, hand_height = self._last_hand_bbox
            palm = np.array([hand_x + hand_width / 2.0, hand_y + hand_height / 2.0])
            ear_points = np.array([[x + width * 0.04, y + height * 0.44], [x + width * 0.96, y + height * 0.44]])
            if float(np.min(np.linalg.norm(ear_points - palm, axis=1))) <= 60.0:
                return "HANDHELD_VIOLATION"
        variation = np.ptp(np.asarray(self.mar_history, dtype=np.float32)) if len(self.mar_history) >= 6 else 0.0
        if has_earphones and not is_yawning and len(self.mar_history) >= 6 and 0.02 <= variation <= 0.3:
            return "HANDS_FREE_CALL"
        return "NONE"

    @staticmethod
    def _gaze_zone(head_yaw: float, head_pitch: float) -> str:
        if head_pitch > 18.0:
            return "LOOKING_DOWN"
        if head_yaw < -22.0:
            return "LEFT_MIRROR"
        if head_yaw > 22.0:
            return "RIGHT_MIRROR"
        return "LOOKING_AHEAD"

    def _detect_face(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.face_detector is not None:
            try:
                self.face_detector.setInputSize((gray.shape[1], gray.shape[0]))
                _, detections = self.face_detector.detect(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                if detections is not None and len(detections):
                    detection = max(detections, key=lambda row: float(row[2] * row[3]))
                    return tuple(int(max(value, 0)) for value in detection[:4])
            except (cv2.error, TypeError, ValueError):
                pass
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
                if len(faces):
                    return tuple(map(int, max(faces, key=lambda item: item[2] * item[3])))
            except (cv2.error, TypeError, ValueError):
                pass
        return None

    @staticmethod
    def _detect_hands(frame: np.ndarray) -> tuple[tuple[int, int, int, int] | None, int, bool]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 220, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) > frame.shape[0] * frame.shape[1] * 0.003]
        if not candidates:
            return None, 0, False
        boxes = [cv2.boundingRect(contour) for contour in sorted(candidates, key=cv2.contourArea, reverse=True)[:2]]
        x_min = min(box[0] for box in boxes)
        y_min = min(box[1] for box in boxes)
        x_max = max(box[0] + box[2] for box in boxes)
        y_max = max(box[1] + box[3] for box in boxes)
        lower_half = any(box[1] + box[3] / 2 >= frame.shape[0] * 0.55 for box in boxes)
        return (x_min, y_min, x_max - x_min, y_max - y_min), len(boxes), lower_half

    def extract(self, frame: np.ndarray) -> DMSFeatures:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("DMS input frame must have shape (height, width, 3)")
        height, width = frame.shape[:2]
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_bbox = self._detect_face(gray)
        if face_bbox is None:
            landmarks = np.empty((0, 2), dtype=np.float32)
            ear = mar = head_yaw = head_pitch = head_roll = 0.0
        else:
            landmarks = self._landmarks_from_contours(gray, face_bbox)
            ear, mar, head_yaw, head_pitch, head_roll = self._face_measurements(gray, face_bbox, landmarks, width, height)
            x, y, box_width, box_height = face_bbox
            cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (80, 230, 180), 2)
            for point in landmarks:
                cv2.circle(annotated, tuple(point.astype(int)), 1, (80, 210, 240), -1)

        hand_bbox, hands_detected, hands_on_bars = self._detect_hands(frame)
        self._last_hand_bbox = hand_bbox
        if hand_bbox is not None:
            x, y, box_width, box_height = hand_bbox
            cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (240, 180, 80), 2)

        has_face = face_bbox is not None
        is_drowsy = bool(has_face and ear < self.ear_threshold)
        is_yawning = bool(has_face and mar > self.mar_threshold)
        if is_yawning and not self.was_yawning:
            print("WARNING: RIDER YAWN DETECTED - FATIGUE RISK")
        self.was_yawning = is_yawning
        has_glasses = bool(has_face and self._detect_glasses(gray, face_bbox, landmarks))
        has_earphones = bool(has_face and self._detect_earphones(gray, face_bbox))
        phone_call_status = self._classify_phone_call(frame, face_bbox, landmarks, mar, has_earphones, is_yawning)
        gaze_zone = self._gaze_zone(head_yaw, head_pitch) if has_face else "UNKNOWN"
        is_distracted = bool(phone_call_status != "NONE" or (has_face and abs(head_yaw) > 22.0) or (hands_detected > 0 and not hands_on_bars))
        self.closed_frames = self.closed_frames + 1 if is_drowsy else max(0, self.closed_frames - 2)
        drowsiness_score = float(np.clip(self.closed_frames / 15.0 * 100.0, 0.0, 100.0)) if has_face else 0.0
        attention_status = "Distracted" if has_face and abs(head_yaw) > 22.0 else "Alert"
        rider_status = "DROWSY" if is_drowsy else "DISTRACTED" if is_distracted or is_yawning else "ALERT"
        if face_bbox:
            face_x, face_y, face_width, face_height = face_bbox
            gauge_x = min(width - 100, max(4, face_x + face_width + 8))
            gauge_y = max(8, face_y + 8)
            for index, (name, value, maximum, color) in enumerate((
                ("EAR", ear, 0.4, (80, 230, 180) if ear >= 0.18 else (68, 23, 255)),
                ("MAR", mar, 1.0, (68, 23, 255) if mar > 0.45 else (255, 229, 0)),
            )):
                y_position = gauge_y + index * 22
                cv2.rectangle(annotated, (gauge_x, y_position), (min(width - 4, gauge_x + 72), y_position + 7), (45, 60, 68), -1)
                fill_width = int(72 * min(1.0, max(0.0, value) / maximum))
                cv2.rectangle(annotated, (gauge_x, y_position), (min(width - 4, gauge_x + fill_width), y_position + 7), color, -1)
                cv2.putText(annotated, f"{name} {value:.2f}", (gauge_x, y_position - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (235, 242, 245), 1, cv2.LINE_AA)
        return DMSFeatures(
            landmarks=landmarks, face_bbox=face_bbox, hand_bbox=hand_bbox, ear=ear, mar=mar,
            hand_position="On Bars" if hands_on_bars else "Off Bars", neck_angle=head_yaw,
            drowsiness_score=drowsiness_score, yawn_detected=is_yawning, attention_status=attention_status,
            rider_status=rider_status, is_drowsy=is_drowsy, is_yawning=is_yawning,
            hands_detected=hands_detected, hands_on_bars=hands_on_bars, head_yaw=head_yaw,
            head_pitch=head_pitch, head_roll=head_roll, annotated_frame=annotated,
            has_glasses=has_glasses, has_earphones=has_earphones,
            phone_call_status=phone_call_status, is_distracted=is_distracted,
            gaze_zone=gaze_zone,
        )

    def close(self) -> None:
        """Keep the same lifecycle API as the former MediaPipe implementation."""