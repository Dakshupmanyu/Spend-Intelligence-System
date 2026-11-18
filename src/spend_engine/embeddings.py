from __future__ import annotations

from typing import Any


MULTILINGUAL_E5_SMALL = "intfloat/multilingual-e5-small"


def ensure_embedding_model_registered(text_embedding_class: Any, model_name: str) -> None:
    """Register multilingual-e5-small in FastEmbed when the installed registry lacks it.

    FastEmbed documents this model as a runtime custom-model registration. The
    registry differs by package release, so check first and only register when
    the requested model is absent.
    """
    requested = model_name.strip().lower()
    supported = text_embedding_class.list_supported_models()
    if any(str(item.get("model", "")).lower() == requested for item in supported):
        return

    if requested != MULTILINGUAL_E5_SMALL:
        raise ValueError(
            f"Embedding model '{model_name}' is not registered in the installed FastEmbed version"
        )

    try:
        from fastembed.common.model_description import ModelSource, PoolingType
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Installed FastEmbed does not expose custom-model registration required for "
            f"{MULTILINGUAL_E5_SMALL}. Upgrade FastEmbed or choose a registered model."
        ) from exc

    text_embedding_class.add_custom_model(
        model=MULTILINGUAL_E5_SMALL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=MULTILINGUAL_E5_SMALL),
        dim=384,
        model_file="onnx/model.onnx",
    )
