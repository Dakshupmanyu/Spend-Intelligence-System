from __future__ import annotations

import sys
from types import ModuleType

import pytest

from spend_engine.embeddings import ensure_embedding_model_registered


class _AlreadySupported:
    @staticmethod
    def list_supported_models():
        return [{"model": "intfloat/multilingual-e5-small"}]

    @staticmethod
    def add_custom_model(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("model already exists")


def test_registered_embedding_model_is_not_registered_twice():
    ensure_embedding_model_registered(_AlreadySupported, "intfloat/multilingual-e5-small")


def test_unknown_model_fails_with_actionable_error():
    with pytest.raises(ValueError, match="not registered"):
        ensure_embedding_model_registered(_AlreadySupported, "not-a-real/model")


def test_multilingual_e5_small_is_registered_when_missing(monkeypatch):
    calls = {}

    class PoolingType:
        MEAN = "mean-pooling"

    class ModelSource:
        def __init__(self, hf):
            self.hf = hf

    common = ModuleType("fastembed.common")
    model_description = ModuleType("fastembed.common.model_description")
    model_description.PoolingType = PoolingType
    model_description.ModelSource = ModelSource
    monkeypatch.setitem(sys.modules, "fastembed.common", common)
    monkeypatch.setitem(sys.modules, "fastembed.common.model_description", model_description)

    class _MissingModel:
        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5"}]

        @staticmethod
        def add_custom_model(**kwargs):
            calls.update(kwargs)

    ensure_embedding_model_registered(_MissingModel, "intfloat/multilingual-e5-small")

    assert calls["model"] == "intfloat/multilingual-e5-small"
    assert calls["dim"] == 384
    assert calls["pooling"] == PoolingType.MEAN
    assert calls["normalization"] is True
    assert calls["sources"].hf == "intfloat/multilingual-e5-small"
    assert calls["model_file"] == "onnx/model.onnx"
