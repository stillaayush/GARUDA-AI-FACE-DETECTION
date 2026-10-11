"""Streamlit command dashboard for GARUDA rider intent and safety analytics."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from api_backend.server import TrajectoryPoint, TrajectoryRequest, service
from models.intent_forecaster import INTENT_NAMES

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows static checks without Streamlit installed
    st = None


def _sample_points() -> list[TrajectoryPoint]:
    timestamps = np.arange(60, dtype=float) / 30.0
    return [TrajectoryPoint(timestamp=float(time), x=float(320 + 1.8 * index), y=float(460 - 0.2 * index), head_yaw=float(10 * np.sin(index / 10))) for index, time in enumerate(timestamps)]


def _load_csv(uploaded: object) -> list[TrajectoryPoint]:
    if uploaded is None:
        return _sample_points()
    frame = pd.read_csv(uploaded)
    required = {"timestamp", "x", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    allowed = {field for field in TrajectoryPoint.model_fields}
    return [TrajectoryPoint(**{key: row[key] for key in allowed if key in row and pd.notna(row[key])}) for _, row in frame.tail(90).iterrows()]


def _render_video_mode(uploaded: object, show_boxes: bool, show_pose: bool) -> None:
    if uploaded is None:
        st.info("Upload an MP4 or AVI clip to start the annotated viewport.")
        return
    st.video(uploaded)
    st.caption("Video preview is ready. Configure YOLO weights in the service layer to stream annotated detections.")
    st.json({"bounding_boxes": show_boxes, "skeletal_pose": show_pose, "detector": "YOLO pose + ByteTrack"})


def _download_csv(result: dict[str, object]) -> bytes:
    forecast = result["forecast"]
    frame = pd.DataFrame(forecast["future_path"], columns=["x", "y"])
    frame["intent"] = forecast["intent"]
    frame["risk"] = result["risk"]["level"]
    return frame.to_csv(index=False).encode("utf-8")


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required. Install dependencies with: pip install -r requirements.txt")
    st.set_page_config(page_title="PROJECT GARUDA", page_icon="G", layout="wide")
    st.title("PROJECT GARUDA")
    st.caption("Multi-agent rider intent forecasting and safety analytics")

    with st.sidebar:
        st.header("Command configuration")
        mode = st.selectbox("Input mode", ["CSV Trajectory Benchmark", "Video Upload (MP4/AVI)", "Real-Time Webcam / RTSP", "SUMO/CARLA Simulation Loader"])
        engine = st.selectbox("Inference engine", ["baseline", "lstm", "transformer"], format_func=lambda value: {"baseline": "XGBoost / Random Forest baseline", "lstm": "Bidirectional LSTM", "transformer": "Temporal Transformer"}[value])
        horizon = st.select_slider("Forecast horizon", options=[1.0, 2.0, 3.0], value=2.0, format_func=lambda value: f"{value:.1f}s")
        show_boxes = st.toggle("Bounding boxes", True)
        show_pose = st.toggle("Skeletal pose", True)
        show_path = st.toggle("Trajectory arrow", True)
        show_heatmap = st.toggle("Risk heatmap", True)
        show_shap = st.toggle("SHAP explainability", True)
        uploaded = st.file_uploader("Trajectory CSV or video", type=["csv", "mp4", "avi"])

    if mode == "Video Upload (MP4/AVI)":
        _render_video_mode(uploaded, show_boxes, show_pose)
        return
    if mode == "Real-Time Webcam / RTSP":
        st.warning("Webcam/RTSP capture is available through the FastAPI detector process; benchmark controls remain active here.")
    if mode == "SUMO/CARLA Simulation Loader":
        st.info("Load a simulator-exported CSV with timestamp, x, and y columns.")

    try:
        points = _load_csv(uploaded if mode != "Real-Time Webcam / RTSP" else None)
        result = service.predict(TrajectoryRequest(history=points, engine=engine, horizon_seconds=horizon))
    except (ValueError, KeyError) as error:
        st.error(str(error))
        return

    forecast = result["forecast"]
    risk = result["risk"]
    summary, path_column, explain_column = st.columns([1.0, 1.5, 1.0])
    with summary:
        st.subheader(forecast["intent"])
        st.metric("Confidence", f"{max(forecast['probabilities'].values()):.1%}")
        st.metric("Risk", risk["level"].upper(), f"{risk['minimum_distance']:.1f}px minimum distance")
        st.caption(f"Rider {forecast['rider_id']} | {forecast['horizon_seconds']:.1f}s horizon | {engine}")
    with path_column:
        st.subheader("Projected path")
        path = pd.DataFrame(forecast["future_path"], columns=["x", "y"])
        path["step"] = np.arange(1, len(path) + 1)
        if show_path:
            st.line_chart(path.set_index("step")[["x", "y"]], height=250)
        if show_heatmap:
            st.progress(float(risk["score"]), text=f"Risk score {risk['score']:.2f} | {risk['reason']}")
    with explain_column:
        st.subheader("Intent probabilities")
        probabilities = pd.DataFrame({"intent": list(forecast["probabilities"]), "probability": list(forecast["probabilities"].values())}).set_index("intent")
        st.bar_chart(probabilities, height=250)
        if show_shap:
            st.dataframe(pd.DataFrame(forecast["attributions"]), hide_index=True, width="stretch")

    st.subheader("Rider telemetry")
    feature_frame = pd.DataFrame([result["features"]]).T.rename(columns={0: "value"})
    st.dataframe(feature_frame, use_container_width=True)
    st.download_button("Export telemetry CSV", _download_csv(result), "garuda_telemetry.csv", "text/csv")
    st.download_button("Export prediction JSON", str(result).encode("utf-8"), "garuda_prediction.json", "application/json")


if __name__ == "__main__":
    main()
