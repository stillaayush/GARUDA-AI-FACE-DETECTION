import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RiderIntentDataset
from models import MultimodalCrossAttentionTransformer


def main() -> None:
    if not Path("data/metadata.csv").exists():
        subprocess.run([sys.executable, "generate_synthetic_data.py"], check=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(RiderIntentDataset(), batch_size=4, shuffle=True, pin_memory=device.type == "cuda")
    model = MultimodalCrossAttentionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    for epoch in range(5):
        model.train()
        correct = total = 0
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/5"):
            video = batch["video"].to(device, non_blocking=True)
            imu = batch["imu"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(video, imu)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / max(total, 1)
        print(f"epoch {epoch + 1}/5 loss={running_loss / max(total, 1):.4f} accuracy={accuracy:.2%}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "checkpoints/best_intent_model.pth")
    print(f"Saved checkpoints/best_intent_model.pth ({best_accuracy:.2%} training accuracy)")


if __name__ == "__main__":
    main()
