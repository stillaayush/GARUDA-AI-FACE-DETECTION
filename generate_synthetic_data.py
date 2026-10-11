import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


LABELS = {"CRUISE_LANE_KEEP": 0, "INTENT_LEFT_TURN": 1, "INTENT_RIGHT_TURN": 2, "INTENT_HARD_BRAKE": 3}


def make_clip(label: str, index: int, frames: int, output: Path) -> dict[str, object]:
    width, height = 160, 120
    video_path = output / "videos" / f"clip_{index:04d}.mp4"
    imu_path = output / "imu" / f"clip_{index:04d}.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    rng = np.random.default_rng(index)
    imu_rows = []
    for frame_index in range(frames):
        progress = frame_index / max(frames - 1, 1)
        image = np.full((height, width, 3), (35, 42, 38), dtype=np.uint8)
        cv2.line(image, (width // 2, 65), (width // 2, height), (120, 125, 115), 3)
        cv2.line(image, (25, 65), (0, height), (70, 78, 72), 2)
        cv2.line(image, (135, 65), (width, height), (70, 78, 72), 2)
        drift = (-26.0 if label == "INTENT_LEFT_TURN" else 26.0 if label == "INTENT_RIGHT_TURN" else 0.0) * progress
        rider_x = int(width / 2 + drift)
        if label == "INTENT_HARD_BRAKE":
            image[:, :, 2] = np.clip(image[:, :, 2].astype(np.int16) + 65, 0, 255).astype(np.uint8)
            cv2.rectangle(image, (rider_x - 30, height - 35), (rider_x + 30, height - 20), (30, 30, 255), -1)
        cv2.circle(image, (rider_x, height - 82), 14, (215, 220, 210), -1)
        cv2.rectangle(image, (rider_x - 20, height - 70), (rider_x + 20, height - 25), (55, 72, 68), -1)
        writer.write(image)
        roll = (-0.35 if label == "INTENT_LEFT_TURN" else 0.35 if label == "INTENT_RIGHT_TURN" else 0.0) * progress
        pitch = 0.65 * progress if label == "INTENT_HARD_BRAKE" else 0.03 * np.sin(progress * 8)
        yaw_rate = (-0.9 if label == "INTENT_LEFT_TURN" else 0.9 if label == "INTENT_RIGHT_TURN" else 0.0) * progress
        ax = -3.8 * progress if label == "INTENT_HARD_BRAKE" else 0.03 * rng.normal()
        imu_rows.append([roll, pitch, yaw_rate, ax, 0.04 * rng.normal(), 9.81 + 0.08 * rng.normal()])
    writer.release()
    pd.DataFrame(imu_rows, columns=["Roll", "Pitch", "Yaw_Rate", "Ax", "Ay", "Az"]).to_csv(imu_path, index=False)
    return {"video": str(video_path.relative_to(output)), "imu": str(imu_path.relative_to(output)), "label": LABELS[label], "label_name": label}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create balanced synchronized GARUDA training data.")
    parser.add_argument("--clips", type=int, default=120)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    output = Path("data")
    (output / "videos").mkdir(parents=True, exist_ok=True)
    (output / "imu").mkdir(parents=True, exist_ok=True)
    names = list(LABELS)
    rows = [make_clip(names[index % len(names)], index, args.frames, output) for index in range(args.clips)]
    pd.DataFrame(rows).to_csv(output / "metadata.csv", index=False)
    print(f"Generated {len(rows)} balanced synchronized clips in {output.resolve()}")


if __name__ == "__main__":
    main()
