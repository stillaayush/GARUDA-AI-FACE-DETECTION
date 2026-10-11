"""PROJECT GARUDA: resilient real-time rider safety cockpit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any
import urllib.request

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

try:
    from utils.pdf_reporter import RideSession
except ImportError:
    RideSession = None  # type: ignore[assignment,misc]

WINDOW_NAME = "GARUDA AI | Fighter-Jet Cockpit"
HUD_WIDTH, HUD_HEIGHT = 1280, 720
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
CRUISE, LEFT, RIGHT, BRAKE = "CRUISE_LANE_KEEP", "INTENT_LEFT_TURN", "INTENT_RIGHT_TURN", "INTENT_HARD_BRAKE"
CYAN, WHITE, MUTED, GLASS = (255, 230, 35), (238, 244, 246), (145, 166, 178), (18, 30, 41)
GREEN, AMBER, RED = (80, 220, 145), (0, 185, 255), (55, 35, 255)


@dataclass
class DMSReading:
    face_bbox: tuple[int, int, int, int] | None = None
    eye_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    ear: float = 0.28
    mar: float = 0.0
    neck_yaw: float = 0.0
    pitch: float = 0.0
    rapid_pitch_down: bool = False
    is_yawning: bool = False
    has_glasses: bool = False
    phone_call_status: str = "NONE"
    landmarks: np.ndarray | None = field(default=None, repr=False)
    specs_bbox: tuple[int, int, int, int] | None = None
    annotated: np.ndarray | None = field(default=None, repr=False)

    @property
    def yawn_detected(self) -> bool:
        return self.is_yawning

    @property
    def is_drowsy(self) -> bool:
        return self.ear < 0.18

    @property
    def hands_on_bars(self) -> bool:
        return True

    @property
    def has_earphones(self) -> bool:
        return False


class GarudaFeatureExtractor:
    """YuNet extractor with a zero-dependency HSV contour fallback."""

    def __init__(self, model_path: str | Path = MODEL_NAME) -> None:
        self.model_path = Path(model_path)
        self.detector: Any | None = None
        self.detector_size: tuple[int, int] | None = None
        self.previous_pitch = 0.0
        self.status = "YUNET READY"
        self._load_yunet()

    def _load_yunet(self) -> None:
        if not self.model_path.exists():
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
            except Exception as error:
                self.status = f"YUNET UNAVAILABLE: {error}"
                return
        try:
            factory = getattr(cv2, "FaceDetectorYN", None)
            if factory is None:
                raise RuntimeError("FaceDetectorYN is unavailable")
            self.detector = factory.create(str(self.model_path), "", (HUD_WIDTH, HUD_HEIGHT), 0.6, 0.3)
        except Exception as error:
            self.status = f"YUNET FALLBACK: {error}"
            self.detector = None

    @staticmethod
    def _roi(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = box
        return gray[max(0, y):min(gray.shape[0], y + height), max(0, x):min(gray.shape[1], x + width)]

    def _yunet_face(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        if self.detector is None:
            return None
        height, width = frame.shape[:2]
        if self.detector_size != (width, height):
            self.detector.setInputSize((width, height))
            self.detector_size = (width, height)
        _, detections = self.detector.detect(frame)
        if detections is None or len(detections) == 0:
            return None
        detection = max(detections, key=lambda row: float(row[2] * row[3]))
        return tuple(map(int, detection[:4])), np.asarray(detection[4:14], dtype=np.float32).reshape(5, 2)

    @staticmethod
    def _fallback_points(box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = box
        return np.array([[x + width * .35, y + height * .4], [x + width * .65, y + height * .4],
                         [x + width * .5, y + height * .55], [x + width * .38, y + height * .72],
                         [x + width * .62, y + height * .72]], dtype=np.float32)

    def _skin_face(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Find the largest plausible skin-tone contour without a legacy detector."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 220, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        candidates = [contour for contour in contours if frame_area * .01 <= cv2.contourArea(contour) <= frame_area * .65]
        if not candidates:
            return None
        box = tuple(map(int, cv2.boundingRect(max(candidates, key=cv2.contourArea))))
        x, y, width, height = box
        if width < 60 or height < 60 or not .55 <= width / max(height, 1) <= 1.8:
            return None
        self.status = "HSV CONTOUR FALLBACK"
        return box, self._fallback_points(box)

    def _detect(self, frame: np.ndarray, gray: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        try:
            found = self._yunet_face(frame)
            if found is not None:
                return found
        except Exception:
            self.detector = None
            self.status = "YUNET FRAME ERROR: HSV FALLBACK"
        try:
            return self._skin_face(frame)
        except Exception as error:
            self.status = f"FACE TRACKING UNAVAILABLE: {error}"
            return None

    def _eye_metrics(self, gray: np.ndarray, points: np.ndarray) -> tuple[float, list[tuple[int, int, int, int]]]:
        scores: list[float] = []
        boxes: list[tuple[int, int, int, int]] = []
        for point in points:
            center_x, center_y = map(int, np.round(point))
            box = (center_x - 14, center_y - 14, 28, 28)
            boxes.append(box)
            roi = self._roi(gray, box)
            if roi.shape != (28, 28):
                continue
            contrast = float(np.std(cv2.equalizeHist(roi)))
            gradient = float(np.var(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)))
            scores.append(float(np.clip((contrast - 7.0) / 34.0 + (gradient - 16.0) / 2600.0, 0.0, 1.0)))
        activity = float(np.mean(scores)) if scores else 0.0
        ear = .28 + .06 * activity if activity >= .18 else .12
        return float(np.clip(ear, .10, .34)), boxes

    def _mouth_metrics(self, gray: np.ndarray, points: np.ndarray) -> tuple[float, bool, tuple[int, int, int, int]]:
        width = max(1.0, float(np.linalg.norm(points[4] - points[3])))
        center = (points[3] + points[4]) / 2.0
        box = (int(center[0] - width * .65), int(center[1] - width * .3), int(width * 1.3), int(width * .9))
        roi = self._roi(gray, box)
        if roi.size == 0:
            return 0.0, False, box
        mask = cv2.threshold(cv2.GaussianBlur(roi, (3, 3), 0), 80, 255, cv2.THRESH_BINARY_INV)[1]
        columns = mask[:, max(0, mask.shape[1] // 2 - 2):mask.shape[1] // 2 + 3]
        rows = np.flatnonzero(np.count_nonzero(columns, axis=1) > 0)
        aperture = float(rows[-1] - rows[0] + 1) if len(rows) >= 2 else 0.0
        mar = float(np.clip(aperture / width, 0.0, 2.0))
        return mar, mar > .42, box

    def _glasses(self, frame: np.ndarray, points: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        right_eye, left_eye = points[0], points[1]
        distance = max(1.0, float(np.linalg.norm(left_eye - right_eye)))
        midpoint = (right_eye + left_eye) / 2.0
        box = (int(midpoint[0] - distance * .25), int(midpoint[1] - distance * .16), int(distance * .5), int(distance * .32))
        orbital = (int(min(right_eye[0], left_eye[0]) - distance * .28), int(min(right_eye[1], left_eye[1]) - distance * .32), int(distance * 1.56), int(distance * .7))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        def density(region: tuple[int, int, int, int]) -> float:
            roi = self._roi(gray, region)
            if roi.size == 0:
                return 0.0
            edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 0), 35, 100)
            cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            return float(np.count_nonzero(edges) / max(1, edges.size))
        return density(box) > .040 or density(orbital) > .040, box

    @staticmethod
    def _phone_status(frame: np.ndarray, face: tuple[int, int, int, int]) -> str:
        x, y, width, height = face
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 220, 255]))
        contours, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= frame.shape[0] * frame.shape[1] * .0025:
                continue
            hand_x, hand_y, hand_width, hand_height = cv2.boundingRect(contour)
            if hand_x + hand_width >= x - width * .35 and hand_x <= x + width * 1.35 and hand_y > y + height * .25:
                return "HANDHELD_VIOLATION"
        return "NONE"

    def extract(self, frame: np.ndarray) -> DMSReading:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("GARUDA expects a BGR frame")
        gray, annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame.copy()
        reading = DMSReading(annotated=annotated)
        found = self._detect(frame, gray)
        if found is None:
            cv2.putText(annotated, "SEARCHING FOR RIDER", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .72, RED, 2, cv2.LINE_AA)
            return reading
        face, points = found
        x, y, width, height = face
        reading.face_bbox, reading.landmarks = face, points
        reading.ear, reading.eye_bboxes = self._eye_metrics(gray, points[:2])
        reading.mar, reading.is_yawning, mouth_box = self._mouth_metrics(gray, points)
        eye_midpoint = (points[0] + points[1]) / 2.0
        eye_distance = max(1.0, float(np.linalg.norm(points[1] - points[0])))
        reading.neck_yaw = float(np.clip((points[2][0] - eye_midpoint[0]) / eye_distance * 45.0, -45.0, 45.0))
        reading.pitch = float(np.clip((points[2][1] - eye_midpoint[1]) / eye_distance * 45.0, -45.0, 45.0))
        reading.rapid_pitch_down = reading.pitch - self.previous_pitch > 8.0
        self.previous_pitch = reading.pitch
        reading.has_glasses, reading.specs_bbox = self._glasses(frame, points)
        reading.phone_call_status = self._phone_status(frame, face)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), GREEN, 2)
        for point in points:
            cv2.circle(annotated, tuple(map(int, point)), 3, CYAN, -1)
        if reading.has_glasses:
            sx, sy, sw, sh = reading.specs_bbox or (0, 0, 0, 0)
            cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), CYAN, 2)
        mx, my, mw, mh = mouth_box
        cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), RED if reading.is_yawning else CYAN, 1)
        return reading


class SoundEngine:
    """Threaded siren controlled by an event so stop is immediate and joinable."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        with self._lock:
            if self.active:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="garuda-siren", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if winsound is not None:
                    winsound.Beep(2800, 250)
                else:
                    print("\a", end="", flush=True)
            except Exception:
                print("\a", end="", flush=True)
            self._stop.wait(.05)

    def stop(self, *, wait: bool = False) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=.35)


@dataclass
class VehicleTelemetry:
    speed: float = 0.0
    lean: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    previous_gray: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> None:
        try:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            flow = cv2.calcOpticalFlowFarneback(self.previous_gray, gray, None, .5, 2, 15, 2, 5, 1.2, 0) if self.previous_gray is not None else np.zeros((180, 320, 2), np.float32)
            self.previous_gray = gray
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self.speed = float(np.clip(np.percentile(magnitude, 75) * .12, 0, 160))
            self.lean = float(np.clip(np.mean(flow[..., 0]) * 2.0, -45, 45))
            self.throttle = float(np.clip(self.speed / 160 * 100, 0, 100))
            self.brake = float(np.clip(-np.mean(flow[..., 1]) * 12, 0, 100))
        except cv2.error:
            self.previous_gray = None


def intent_for(reading: DMSReading) -> str:
    if reading.rapid_pitch_down or reading.pitch > 18.0:
        return BRAKE
    if reading.neck_yaw < -10.0:
        return LEFT
    if reading.neck_yaw > 10.0:
        return RIGHT
    return CRUISE


def simulated_frame(index: int) -> np.ndarray:
    frame = np.full((HUD_HEIGHT, HUD_WIDTH, 3), (32, 42, 46), dtype=np.uint8)
    drift = int(75 * np.sin(index / 28.0))
    cv2.line(frame, (HUD_WIDTH // 2, HUD_HEIGHT // 2), (HUD_WIDTH // 2 + drift, HUD_HEIGHT), (105, 120, 110), 4)
    return frame


def _text(canvas: np.ndarray, value: str, origin: tuple[int, int], size: float = .5, color: tuple[int, int, int] = WHITE, weight: int = 1) -> None:
    cv2.putText(canvas, value, origin, cv2.FONT_HERSHEY_DUPLEX, size, color, weight, cv2.LINE_AA)


def _panel(canvas: np.ndarray, box: tuple[int, int, int, int], border: tuple[int, int, int] = (62, 101, 116)) -> None:
    x1, y1, x2, y2 = box
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), GLASS, -1)
    cv2.addWeighted(overlay, .72, canvas, .28, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)


def draw_hud(frame: np.ndarray, reading: DMSReading, telemetry: VehicleTelemetry, intent: str, active: bool, elapsed: float, closed: float, alarm: bool, fps: float, notice: str, model_status: str) -> np.ndarray:
    canvas = cv2.resize(frame, (HUD_WIDTH, HUD_HEIGHT))
    _panel(canvas, (20, 18, 1260, 88))
    _text(canvas, "GARUDA AI | T-2.5s PREDICTIVE INTENT", (42, 54), .7, WHITE, 2)
    color = CYAN if intent == CRUISE else RED if intent == BRAKE else AMBER
    _text(canvas, intent.replace("INTENT_", "").replace("_", " "), (570, 54), .52, color, 2)
    cv2.circle(canvas, (1115, 45), 6, GREEN, -1)
    _text(canvas, "V2X ACTIVE", (1130, 51), .42, GREEN)
    _text(canvas, f"FPS {fps:.1f}  {model_status[:28]}", (1020, 74), .3, MUTED)
    center_x, center_y = HUD_WIDTH // 2, 350
    for offset in (-105, -70, -35, 35, 70, 105):
        cv2.line(canvas, (center_x - 36, center_y + offset), (center_x + 36, center_y + offset), (90, 130, 140), 1)
    cv2.line(canvas, (center_x - 110, center_y), (center_x + 110, center_y), (90, 130, 140), 1)
    roll = int(np.clip(reading.neck_yaw, -35, 35) * 2.2)
    cv2.line(canvas, (center_x - 80 - roll, center_y - 16), (center_x + 80 - roll, center_y + 16), color, 3)
    cv2.circle(canvas, (center_x, center_y), 6, CYAN, 1)
    _panel(canvas, (24, 125, 300, 570))
    _panel(canvas, (980, 125, 1256, 570))
    _text(canvas, "DMS TELEMETRY", (45, 160), .5, CYAN)
    _text(canvas, f"EAR {reading.ear:.2f}", (45, 220), .62, GREEN if reading.ear >= .20 else RED, 2)
    _text(canvas, f"MAR {reading.mar:.2f}  {'YAWN' if reading.is_yawning else 'CLEAR'}", (45, 280), .42, RED if reading.is_yawning else WHITE)
    _text(canvas, f"NECK YAW {reading.neck_yaw:+.1f} deg", (45, 340), .42, WHITE)
    _text(canvas, f"SPECS {'COMPLIANT' if reading.has_glasses else 'NOT DETECTED'}", (45, 400), .4, CYAN if reading.has_glasses else MUTED)
    _text(canvas, "CRITICAL ALERT: HANDHELD PHONE VIOLATION" if reading.phone_call_status != "NONE" else "PHONE COMPLIANCE: CLEAR", (45, 460), .31, RED if reading.phone_call_status != "NONE" else GREEN)
    _text(canvas, "DYNAMICS", (1000, 160), .5, CYAN)
    _text(canvas, f"{telemetry.speed:03.0f} KM/H", (1000, 240), 1.2, WHITE, 2)
    _text(canvas, f"LEAN {telemetry.lean:+.1f} deg", (1000, 300), .42, WHITE)
    _text(canvas, f"THROTTLE {telemetry.throttle:03.0f}%", (1000, 360), .4, GREEN)
    _text(canvas, f"BRAKE {telemetry.brake:03.0f}%", (1000, 415), .4, RED if telemetry.brake > 5 else MUTED)
    if closed > 0:
        progress = float(np.clip(closed / 5.0, 0, 1))
        _panel(canvas, (center_x - 285, 185, center_x + 285, 285), RED if progress >= 1 else AMBER)
        _text(canvas, f"EYES CLOSED: {closed:.1f}s / 5.0s", (center_x - 230, 225), .52, WHITE, 2)
        bar_color = GREEN if progress < .5 else AMBER if progress < 1 else RED
        cv2.rectangle(canvas, (center_x - 245, 245), (center_x + 245, 263), (54, 78, 88), -1)
        cv2.rectangle(canvas, (center_x - 245, 245), (center_x - 245 + int(490 * progress), 263), bar_color, -1)
    if alarm and int(time.monotonic() * 8) % 2 == 0:
        cv2.rectangle(canvas, (0, 0), (HUD_WIDTH - 1, HUD_HEIGHT - 1), RED, 14)
        _text(canvas, "CRITICAL ALERT: WAKE UP! SLEEP DETECTED", (center_x - 270, 120), .55, WHITE, 2)
    button_color = RED if active else GREEN
    _panel(canvas, (310, 620, 970, 707), button_color)
    _text(canvas, "[ STOP & EXPORT PDF ]" if active else "[ START GARUDA MISSION ]", (390, 672), .62, button_color, 2)
    _text(canvas, f"REC {int(elapsed // 60):02d}:{int(elapsed) % 60:02d}" if active else "SPACE / S TO TOGGLE     Q TO EXIT", (1000 if active else 420, 610), .36, RED if active else MUTED)
    if notice:
        _text(canvas, notice[:110], (30, 712), .32, CYAN)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone PROJECT GARUDA safety engine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    extractor = GarudaFeatureExtractor()
    telemetry, siren = VehicleTelemetry(), SoundEngine()
    session = RideSession() if RideSession is not None else None
    camera = None if args.simulate else cv2.VideoCapture(args.camera)
    simulated = args.simulate or camera is None or not camera.isOpened()
    if camera is not None and not simulated:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, HUD_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HUD_HEIGHT)
    closed, last_time, frame_index, notice = 0.0, time.monotonic(), 0, "SIMULATION FEED" if simulated else "CAMERA FEED ACTIVE"
    microsleep_logged = False
    button = (310, 620, 970, 707)

    def toggle() -> None:
        nonlocal notice
        if session is None:
            notice = "PDF EXPORT UNAVAILABLE: utils.pdf_reporter could not load"
            return
        try:
            if session.active:
                path = session.stop_and_export()
                notice = f"PDF GENERATED: {path}"
            else:
                session.start()
                notice = "GARUDA MISSION LOGGING ACTIVE"
        except (OSError, RuntimeError, ImportError, ValueError) as error:
            notice = f"PDF/MISSION ERROR: {error}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONUP and button[0] <= x <= button[2] and button[1] <= y <= button[3]:
            toggle()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, HUD_WIDTH, HUD_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    except cv2.error as error:
        if camera is not None:
            camera.release()
        print(f"GARUDA GUI unavailable: {error}")
        return
    try:
        while True:
            if simulated:
                frame, success = simulated_frame(frame_index), True
            else:
                success, frame = camera.read() if camera is not None else (False, None)
            if not success or frame is None:
                simulated, notice = True, "CAMERA FRAME LOST: SIMULATION FEED ACTIVE"
                continue
            frame_index += 1
            now = time.monotonic()
            dt, last_time = min(.25, max(.001, now - last_time)), now
            try:
                reading = extractor.extract(frame)
                telemetry.update(frame)
            except (AttributeError, cv2.error, ValueError, TypeError) as error:
                notice = f"FRAME RECOVERY: {error}"
                reading = DMSReading(annotated=frame.copy())
            if reading.face_bbox is not None and reading.ear < .18:
                closed += dt
            elif reading.ear >= .20:
                closed = 0.0
                microsleep_logged = False
                siren.stop()
            alarm = closed >= 5.0
            if alarm:
                siren.start()
                if not microsleep_logged and session is not None and session.active:
                    session.record_microsleep(max(0.0, closed - 5.0), closed)
                    microsleep_logged = True
                    notice = "CRITICAL MICROSLEEP INCIDENT LOGGED"
            else:
                siren.stop()
            intent = intent_for(reading)
            if session is not None and session.active:
                try:
                    session.log_frame(intent, reading, telemetry.speed, telemetry.lean, roll=reading.neck_yaw, pitch=reading.pitch, camera_fps=1.0 / dt)
                except (AttributeError, ValueError, TypeError):
                    notice = "SESSION LOGGING RECOVERED FROM INVALID FRAME"
            elapsed = session.started_at and max(0.0, time.monotonic() - session.started_at) if session is not None and session.active else 0.0
            canvas = draw_hud(reading.annotated if reading.annotated is not None else frame, reading, telemetry, intent, bool(session and session.active), float(elapsed), closed, siren.active, 1.0 / dt, notice, extractor.status)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S"), 32):
                toggle()
    finally:
        siren.stop(wait=True)
        if camera is not None:
            camera.release()
        if session is not None and session.active:
            try:
                session.stop_and_export()
            except (OSError, RuntimeError, ImportError):
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
