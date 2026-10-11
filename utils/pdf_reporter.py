from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from platform import system
from time import monotonic
from uuid import uuid4

REPORT_TITLE = "PROJECT GARUDA — TELEMETRIC FLIGHT & RIDE SAFETY AUDIT"


def _set_pdf_metadata(canvas, _document) -> None:
    canvas.setTitle(REPORT_TITLE)
    canvas.setAuthor("PROJECT GARUDA")


@dataclass
class RideSession:
    """Collects GARUDA ride telemetry and produces a self-contained PDF summary."""

    started_at: float | None = None
    started_datetime: datetime | None = None
    frames: list[dict] = field(default_factory=list)
    events: list[tuple[float, str]] = field(default_factory=list)
    yawn_count: int = 0
    was_yawning: bool = False
    hands_off_seconds: float = 0.0
    last_frame_at: float | None = None
    frame_count: int = 0
    last_phone_status: str = "NONE"
    microsleep_incidents: list[dict[str, float | str]] = field(default_factory=list)
    incident_log: list[dict[str, str | float]] = field(default_factory=list)
    hands_on_frames: int = 0
    peak_deceleration_g: float = 0.0
    active: bool = False

    def start(self) -> None:
        self.started_at = monotonic()
        self.started_datetime = datetime.now()
        self.frames.clear()
        self.events.clear()
        self.yawn_count = 0
        self.was_yawning = False
        self.hands_off_seconds = 0.0
        self.last_frame_at = None
        self.frame_count = 0
        self.last_phone_status = "NONE"
        self.microsleep_incidents.clear()
        self.incident_log.clear()
        self.hands_on_frames = 0
        self.peak_deceleration_g = 0.0
        self.active = True

    def record_microsleep(self, start_seconds: float, duration: float) -> None:
        incident = {"timestamp": float(start_seconds), "duration": float(duration), "event": "CRITICAL_MICROSLEEP_INCIDENT"}
        self.microsleep_incidents.append(incident)
        self.events.append((float(start_seconds), f"CRITICAL_MICROSLEEP_INCIDENT ({duration:.1f}s)"))
        self.incident_log.append({"timestamp": float(start_seconds), "event": "Micro-sleep Warning", "severity": "CRITICAL", "action": "Emergency Acoustic Siren Fired"})

    def log_frame(self, intent: str, dms, speed: float, lean: float, deceleration_g: float = 0.0, *, roll: float = 0.0, pitch: float = 0.0, gaze_zone: str = "UNKNOWN", camera_fps: float = 0.0, inference_ms: float = 0.0, processing_ms: float = 0.0) -> None:
        if not self.active or self.started_at is None:
            return
        now = monotonic()
        elapsed = now - self.started_at
        frame_delta_seconds = max(0.0, now - self.last_frame_at) if self.last_frame_at is not None else 0.0
        if self.last_frame_at is not None and not dms.hands_on_bars:
            self.hands_off_seconds += frame_delta_seconds
        self.last_frame_at = now
        self.hands_on_frames += int(bool(dms.hands_on_bars))
        self.peak_deceleration_g = max(self.peak_deceleration_g, abs(float(deceleration_g)))
        if dms.yawn_detected and not self.was_yawning:
            self.yawn_count += 1
            self.incident_log.append({"timestamp": elapsed, "event": "Rider Yawn Detected", "severity": "MEDIUM", "action": "Fatigue risk warning displayed"})
        self.was_yawning = bool(dms.yawn_detected)
        if dms.is_drowsy and (not self.events or self.events[-1][1] != "Severe Drowsiness Detected"):
            self.events.append((elapsed, "Severe Drowsiness Detected"))
            self.incident_log.append({"timestamp": elapsed, "event": "Drowsiness Warning", "severity": "HIGH", "action": "Rider alert raised"})
        phone_call_status = getattr(dms, "phone_call_status", "NONE")
        if phone_call_status == "HANDHELD_VIOLATION" and self.last_phone_status != "HANDHELD_VIOLATION":
            self.events.append((elapsed, "Handheld Phone Call Violation"))
            self.incident_log.append({"timestamp": elapsed, "event": "Handheld Phone Call", "severity": "HIGH", "action": "Compliance violation flagged"})
        self.last_phone_status = phone_call_status
        self.frames.append({"intent": intent, "speed": float(speed), "lean": float(lean), "elapsed": elapsed, "dt": frame_delta_seconds})
        self.frames[-1].update({"has_glasses": bool(getattr(dms, "has_glasses", False)), "has_earphones": bool(getattr(dms, "has_earphones", False)), "phone_call_status": phone_call_status, "roll": float(roll), "pitch": float(pitch), "gaze_zone": gaze_zone, "camera_fps": float(camera_fps), "inference_ms": float(inference_ms), "processing_ms": float(processing_ms)})
        self.frame_count += 1
        if intent == "INTENT_HARD_BRAKE" and (not self.events or self.events[-1][1] != "Hard Brake Intent Detected"):
            self.events.append((elapsed, "Hard Brake Intent Detected"))
            self.incident_log.append({"timestamp": elapsed, "event": "Hard Emergency Brake", "severity": "HIGH", "action": "Intent broadcast issued"})

    def stop_and_export(self, output_dir: str | Path = "reports") -> Path:
        if not self.active or self.started_at is None:
            raise RuntimeError("No active ride session")
        duration = max(0.0, monotonic() - self.started_at)
        self.active = False
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_id = f"GARUDA-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6].upper()}"
        report_path = output_path / f"GARUDA_REPORT_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError as error:
            raise RuntimeError("reportlab is required to export PDF reports") from error

        intent_counts = Counter(frame["intent"] for frame in self.frames)
        total = max(1, len(self.frames))
        logged_seconds = sum(frame.get("dt", 0.0) for frame in self.frames)
        average_speed = sum(frame["speed"] * frame.get("dt", 0.0) for frame in self.frames) / logged_seconds if logged_seconds > 0.0 else sum(frame["speed"] for frame in self.frames) / total
        max_speed = max((frame["speed"] for frame in self.frames), default=0.0)
        peak_lean = max((abs(frame["lean"]) for frame in self.frames), default=0.0)
        drowsy_events = sum("Drowsiness" in message for _, message in self.events)
        hard_brake_events = sum("Brake" in message for _, message in self.events)
        phone_violations = sum("Handheld Phone" in message for _, message in self.events)
        glasses_frames = sum(frame["has_glasses"] for frame in self.frames)
        earphone_frames = sum(frame["has_earphones"] for frame in self.frames)
        penalties = min(100.0, len(self.microsleep_incidents) * 15.0 + self.yawn_count * 5.0 + phone_violations * 10.0)
        alertness_score = max(0.0, 100.0 - penalties)
        maneuver_breakdown = {
            label: intent_counts[label] / total
            for label in ("CRUISE_LANE_KEEP", "INTENT_LEFT_TURN", "INTENT_RIGHT_TURN", "INTENT_HARD_BRAKE")
        }
        duration_text = f"{int(duration // 60):02d}:{int(duration) % 60:02d}"
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("GarudaTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#123047"), fontSize=19, leading=23)
        section_style = ParagraphStyle("GarudaSection", parent=styles["Heading2"], textColor=colors.HexColor("#0b7189"), spaceBefore=12, spaceAfter=7)
        footer_style = ParagraphStyle("GarudaFooter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=8, textColor=colors.HexColor("#637684"))
        story = [Paragraph(REPORT_TITLE, title_style), Spacer(1, 0.12 * inch)]
        story.append(Paragraph(f"System: {REPORT_TITLE} &nbsp;&nbsp; Session UUID: {report_id} &nbsp;&nbsp; Date: {datetime.now():%Y-%m-%d %H:%M:%S} &nbsp;&nbsp; Environment: BENCHMARK / LAB TEST ENVIRONMENT &nbsp;&nbsp; Operator: GARUDA Edge Operator / {system()}", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))
        score_grade = "A" if alertness_score >= 90 else "B" if alertness_score >= 80 else "C" if alertness_score >= 70 else "D" if alertness_score >= 60 else "F"
        story.append(Paragraph("SECTION 1 - EXECUTIVE SAFETY SCORECARD", section_style))
        summary = [
            ["Mission Duration", duration_text],
            ["Safety Score", f"{alertness_score:.0f}% (Grade {score_grade})"],
            ["Peak Lean Angle", f"{peak_lean:.1f} deg"],
            ["Cruise Maneuver", f"{maneuver_breakdown['CRUISE_LANE_KEEP']:.1%}"],
            ["Left Turn Maneuver", f"{maneuver_breakdown['INTENT_LEFT_TURN']:.1%}"],
            ["Right Turn Maneuver", f"{maneuver_breakdown['INTENT_RIGHT_TURN']:.1%}"],
            ["Hard Brake Maneuver", f"{maneuver_breakdown['INTENT_HARD_BRAKE']:.1%}"],
            ["Average Speed", f"{average_speed:.1f} km/h"],
            ["Max Speed", f"{max_speed:.1f} km/h"],
            ["Peak Deceleration", f"{self.peak_deceleration_g:.2f} G"],
        ]
        table = Table(summary, colWidths=[2.2 * inch, 3.6 * inch])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f6")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa8b4")), ("PADDING", (0, 0), (-1, -1), 7)]))
        story.extend([table, Spacer(1, 0.12 * inch), Paragraph("SECTION 2 - FATIGUE &amp; BIOMETRIC ANALYSIS", section_style)])
        fatigue = [
            ["Total yawns logged", str(self.yawn_count)],
            ["Extended eye closure events", str(len(self.microsleep_incidents))],
            ["Hands-off-bars duration", f"{self.hands_off_seconds:.1f} sec"],
            ["Hands-on-bars compliance", f"{self.hands_on_frames / total:.1%}"],
            ["Specs Compliance Duration", f"Detected in {glasses_frames / total:.1%} of frames"],
            ["Earphone compliance log", f"Detected in {earphone_frames / total:.1%} of frames"],
            ["Handheld call violations", str(phone_violations)],
        ]
        story.append(Table(fatigue, colWidths=[2.6 * inch, 3.2 * inch], style=[("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f6")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa8b4")), ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f8f9")]), ("PADDING", (0, 0), (-1, -1), 6)]))
        gaze_seconds = Counter()
        for frame in self.frames:
            gaze_seconds[frame.get("gaze_zone", "UNKNOWN")] += frame.get("dt", 0.0)
        gaze_total = sum(gaze_seconds.values()) or 1.0
        gaze_rows = [["Head pose / distraction", "Time share"], *[[label.replace("_", " "), f"{gaze_seconds[label] / gaze_total:.1%}"] for label in ("LOOKING_AHEAD", "LEFT_MIRROR", "RIGHT_MIRROR", "LOOKING_DOWN")]]
        story.append(Spacer(1, 0.08 * inch))
        story.append(Table(gaze_rows, colWidths=[3.8 * inch, 2.0 * inch], style=[("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceff2")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8f9")]), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("PADDING", (0, 0), (-1, -1), 5)]))
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("SECTION 3 - INTENT &amp; MANEUVER FORECASTING DISTRIBUTION", section_style))
        distribution = []
        for label, display in (("CRUISE_LANE_KEEP", "Cruising"), ("INTENT_LEFT_TURN", "Left turn"), ("INTENT_RIGHT_TURN", "Right turn"), ("INTENT_HARD_BRAKE", "Braking")):
            share = intent_counts[label] / total
            bar = Table([["", ""]], colWidths=[max(0.05, 3.1 * share) * inch, max(0.05, 3.1 * (1.0 - share)) * inch], rowHeights=[0.14 * inch])
            bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#0b9db2")), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#d9e4e8")), ("BOX", (0, 0), (-1, -1), 0.2, colors.HexColor("#b2c4ca"))]))
            distribution.append([display, f"{share:.1%}", bar])
        story.append(Table(distribution, colWidths=[1.4 * inch, 0.7 * inch, 3.7 * inch], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("PADDING", (0, 0), (-1, -1), 5)]))
        story.extend([Spacer(1, 0.12 * inch), Paragraph("SECTION 4 - FULL CHRONOLOGICAL INCIDENT LOG", section_style)])
        incidents = [[f"{int(float(item['timestamp']) // 3600):02d}:{int(float(item['timestamp']) // 60) % 60:02d}:{int(float(item['timestamp']) % 60):02d}", str(item["event"]), str(item["severity"]), str(item["action"])] for item in sorted(self.incident_log, key=lambda item: float(item["timestamp"]))]
        incidents = incidents or [["--:--:--", "NONE", "INFO", "Zero safety incidents recorded during this session."]]
        story.append(Table([["Timestamp", "Event Type", "Severity", "Action Taken"], *incidents], colWidths=[0.9 * inch, 1.55 * inch, 0.75 * inch, 2.6 * inch], repeatRows=1, style=[("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8f9")]), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("PADDING", (0, 0), (-1, -1), 5), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.extend([Spacer(1, 0.3 * inch), Paragraph("Generated by GARUDA Multimodal Edge AI System", footer_style)])
        document = SimpleDocTemplate(str(report_path), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
        document.build(story, onFirstPage=_set_pdf_metadata)
        return report_path