"""Offline model evaluation and edge-case trajectory generation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

LABELS = {0: "Straight", 1: "Left Turn", 2: "Right Turn", 3: "Stop / Brake", 4: "Lane Change"}


def evaluate_predictions(y_true: Iterable[int], predictions: dict[str, Iterable[int]], output_dir: str | Path | None = None) -> dict[str, dict[str, object]]:
    """Return accuracy, precision, recall, F1, and optionally save confusion matrices."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    results: dict[str, dict[str, object]] = {}
    output = Path(output_dir) if output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(list(y_true), dtype=np.int64)
    for name, values in predictions.items():
        predicted = np.asarray(list(values), dtype=np.int64)
        if len(predicted) != len(truth):
            raise ValueError(f"{name} predictions are not aligned with y_true")
        report = classification_report(truth, predicted, labels=list(LABELS), target_names=list(LABELS.values()), output_dict=True, zero_division=0)
        matrix = confusion_matrix(truth, predicted, labels=list(LABELS))
        results[name] = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
        }
        if output:
            pd.DataFrame(matrix, index=LABELS.values(), columns=LABELS.values()).to_csv(output / f"{name.lower()}_confusion_matrix.csv")
    if output and predictions:
        try:
            import plotly.express as px
            for name, result in results.items():
                matrix = np.asarray(result["confusion_matrix"])
                figure = px.imshow(matrix, x=list(LABELS.values()), y=list(LABELS.values()), text_auto=True, title=f"{name} confusion matrix")
                figure.write_html(output / f"{name.lower()}_confusion_matrix.html")
        except ImportError:
            pass
    return results


def generate_synthetic_scenarios(count_per_scenario: int = 25, frames: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate SUMO/CARLA-style trajectories for turns, cut-ins, swerves, and braking."""
    if count_per_scenario < 1 or frames < 2:
        raise ValueError("count_per_scenario must be positive and frames must be >= 2")
    rng = np.random.default_rng(seed)
    scenarios = ("straight", "left_turn", "right_turn", "abrupt_cut_in", "emergency_swerve", "intersection_turn")
    rows: list[dict[str, float | int | str]] = []
    for scenario in scenarios:
        for scenario_index in range(count_per_scenario):
            rider_id = len(rows)
            x, y = rng.normal(320, 3), rng.normal(460, 3)
            for frame in range(frames):
                progress = frame / max(frames - 1, 1)
                dx, dy = 2.0, -0.8
                if scenario in {"left_turn", "intersection_turn"}:
                    dx += -3.5 * progress
                elif scenario == "right_turn":
                    dx += 3.5 * progress
                elif scenario == "abrupt_cut_in":
                    dx += 18.0 if progress > 0.65 else 0.0
                elif scenario == "emergency_swerve":
                    dx += 11.0 * np.sin(progress * np.pi * 4)
                    dy += 3.0 * np.sin(progress * np.pi * 2)
                if scenario == "emergency_swerve" and progress > 0.7:
                    dy *= 0.35
                rows.append({"scenario": scenario, "rider_id": rider_id, "frame": frame, "timestamp": frame / 30.0, "x": x, "y": y, "head_yaw": dx * 2.0, "torso_lean": dx * 0.8, "arm_signal": float(scenario in {"left_turn", "right_turn"})})
                x += dx + rng.normal(0, 0.25)
                y += dy + rng.normal(0, 0.15)
    return pd.DataFrame(rows)


def write_synthetic_scenarios(path: str | Path, **kwargs: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_scenarios(**kwargs).to_csv(output, index=False)
    return output
