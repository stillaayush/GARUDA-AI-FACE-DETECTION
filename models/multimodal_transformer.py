"""GARUDA Multimodal Transformer for rider intent and status prediction."""

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class IMUEncoder(nn.Module):
    """Encode 30 six-axis IMU samples into 15 temporal tokens."""

    def __init__(self, d_model: int = 96) -> None:
        super().__init__()
        hidden = max(d_model // 2, 8)
        self.network = nn.Sequential(
            nn.Conv1d(6, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(15),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.ndim != 3 or imu.shape[-1] != 6:
            raise ValueError("IMU input must have shape (batch, 30, 6)")
        return self.network(imu.transpose(1, 2)).transpose(1, 2)


class HandLocalizationHead(nn.Module):
    """Small local heatmap head for hand-region localization."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(image))


class VisionEncoder(nn.Module):
    """Extract one MobileNet feature vector per frame in a 15-frame clip."""

    def __init__(self, d_model: int = 96) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        feature_size = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(feature_size, d_model)
        self.hand_localization_head = HandLocalizationHead()
        self.hand_projection = nn.Linear(1, d_model)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("Video input must have shape (batch, 15, 3, height, width)")
        batch, steps, channels, height, width = video.shape
        images = video.reshape(batch * steps, channels, height, width)
        features = self.projection(self.backbone(images))
        hand_score = self.hand_localization_head(images).mean(dim=(2, 3))
        features = features + self.hand_projection(hand_score)
        return features.reshape(batch, steps, -1)


class RiderStatusEncoder(nn.Module):
    """Encode EAR, MAR, hand position, and neck angle into status tokens."""

    def __init__(self, input_features: int = 4, d_model: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, features: torch.Tensor, steps: int) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(1).expand(-1, steps, -1)
        if features.ndim != 3 or features.shape[-1] != 4:
            raise ValueError("Rider status input must have shape (batch, 4) or (batch, steps, 4)")
        return self.network(features)


class MultimodalCrossAttentionTransformer(nn.Module):
    """GARUDA Multimodal Transformer that fuses vision, IMU, and rider status features."""

    def __init__(self, d_model: int = 96, num_classes: int = 4) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(d_model)
        self.imu_encoder = IMUEncoder(d_model)
        self.rider_status_encoder = RiderStatusEncoder(d_model=d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.status_cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.status_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.status_classifier = nn.Linear(d_model, 3)

    def forward(
        self, video: torch.Tensor, imu: torch.Tensor, rider_status: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_tokens = self.vision_encoder(video)
        imu_tokens = self.imu_encoder(imu)
        attended, attention_weights = self.cross_attention(
            query=vision_tokens, key=imu_tokens, value=imu_tokens,
            need_weights=True, average_attn_weights=True,
        )
        fused = self.norm(vision_tokens + attended)
        if rider_status is None:
            rider_status = torch.zeros(video.shape[0], 4, device=video.device, dtype=video.dtype)
        status_tokens = self.rider_status_encoder(rider_status, fused.shape[1])
        status_attended, _ = self.status_cross_attention(query=status_tokens, key=fused, value=fused, need_weights=False)
        status_fused = self.status_norm(status_tokens + status_attended)
        intent_logits = self.classifier((fused + status_fused).mean(dim=1))
        status_logits = self.status_classifier(status_fused.mean(dim=1))
        return intent_logits, status_logits, attention_weights


class MultimodalIntentModel(MultimodalCrossAttentionTransformer):
    """Backward-compatible name for older scripts."""
