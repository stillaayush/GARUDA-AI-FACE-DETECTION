"""GARUDA Synchronized Dataset for video and six-axis IMU training windows."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RiderIntentDataset(Dataset):
    """Load GARUDA synchronized 15-frame video and 30-row six-axis IMU windows."""

    def __init__(self, metadata_path: str = "data/metadata.csv", frames_per_clip: int = 15, imu_steps: int = 30, image_size: tuple[int, int] = (112, 112)):
        self.root = Path(metadata_path).parent
        self.metadata = pd.read_csv(metadata_path)
        self.frames_per_clip = frames_per_clip
        self.imu_steps = imu_steps
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[index]
        capture = cv2.VideoCapture(str(self.root / row.video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        positions = np.linspace(0, max(total - 1, 0), self.frames_per_clip).astype(int) if total else np.zeros(self.frames_per_clip, dtype=int)
        frames = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            success, frame = capture.read()
            if not success:
                frame = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            frame = cv2.resize(frame, self.image_size)
            frames.append(frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
        capture.release()
        imu_path = self.root / str(row.imu)
        imu = pd.read_csv(imu_path).to_numpy(dtype=np.float32) if imu_path.suffix.lower() == ".csv" else np.load(imu_path).astype(np.float32)
        imu = np.asarray(imu, dtype=np.float32)
        if imu.ndim != 2 or imu.shape[1] < 6:
            raise ValueError(f"IMU file must contain at least six columns: {imu_path}")
        imu = imu[:, :6]
        if len(imu) == 0:
            imu = np.zeros((1, 6), dtype=np.float32)
        imu_positions = np.linspace(0, len(imu) - 1, self.imu_steps).astype(int)
        return {
            "video": torch.tensor(np.asarray(frames), dtype=torch.float32),
            "imu": torch.tensor(imu[imu_positions], dtype=torch.float32),
            "label": torch.tensor(int(row.label), dtype=torch.long),
        }
