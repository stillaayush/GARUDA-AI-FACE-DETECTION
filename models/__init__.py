"""Model exports with lazy loading for optional TorchVision dependencies."""

__all__ = [
	"IMUEncoder",
	"HandLocalizationHead",
	"VisionEncoder",
	"MultimodalCrossAttentionTransformer",
	"MultimodalIntentModel",
	"RiderStatusEncoder",
]


def __getattr__(name: str):
	if name in __all__:
		from . import multimodal_transformer
		return getattr(multimodal_transformer, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
