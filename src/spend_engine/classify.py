from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .embeddings import ensure_embedding_model_registered
from .features import build_semantic_text
from .normalize import normalize_text, token_similarity

LOGGER = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: dict[str, Any], config_dir: Path):
        self.enabled = bool(config.get("use_embeddings", True))
        self.model = None
        self.error = ""
        self.query_prefix = str(config.get("query_prefix", ""))
        self.passage_prefix = str(config.get("passage_prefix", ""))
        self.batch_size = int(config.get("batch_size", 128))
        model_cache = Path(config.get("model_cache_dir", "../.cache_models"))
        if not model_cache.is_absolute():
            model_cache = (config_dir / model_cache).resolve()
        self.cache_dir = model_cache
        self.allow_fallback = bool(config.get("fallback_to_lexical", True))
        if self.enabled:
            try:
                from fastembed import TextEmbedding

                model_name = str(config.get("model_name", "intfloat/multilingual-e5-small"))
                ensure_embedding_model_registered(TextEmbedding, model_name)
                # Cache first (works offline once warmed), then allow a one-time download.
                # A fresh machine therefore never fails just because the cache is empty.
                last_exc: Exception | None = None
                for cache_only in (True, False):
                    try:
                        self.model = TextEmbedding(
                            model_name=model_name,
                            cache_dir=str(self.cache_dir),
                            threads=int(config.get("threads", 2)),
                            cuda=False,
                            local_files_only=cache_only,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                if self.model is None and last_exc is not None:
                    raise last_exc
            except Exception as exc:
                self.model = None
                self.error = str(exc)
                if not self.allow_fallback:
                    raise RuntimeError(f"Could not initialize embedding model: {exc}") from exc
                LOGGER.warning("Embedding model unavailable; lexical fallback enabled: %s", exc)

    @property
    def available(self) -> bool:
        return self.model is not None

    def encode(self, texts: list[str], query: bool = True) -> np.ndarray | None:
        if not self.available:
            return None
        prefix = self.query_prefix if query else self.passage_prefix
        prepared = [prefix + (text or "") for text in texts]
        try:
            vectors = [
                np.asarray(vector, dtype=np.float32)
                for vector in self.model.embed(prepared, batch_size=self.batch_size)
            ]
        except Exception as exc:
            # A failure while encoding must degrade to lexical mode, not abort the run.
            self.error = f"encoding failed: {exc}"
            self.model = None
            if not self.allow_fallback:
                raise RuntimeError(f"Embedding encode failed: {exc}") from exc
            LOGGER.warning("Embedding encode failed; lexical fallback enabled: %s", exc)
            return None
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def _lexical_tokens(value: str) -> set[str]:
    tokens = set()
    for token in normalize_text(value).split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token:
            tokens.add(token)
    return tokens


def _cosine_score(query_vector: np.ndarray | None, candidate_vector: np.ndarray | None) -> float:
    if query_vector is None or candidate_vector is None:
        return 0.0
    return float(np.dot(query_vector, candidate_vector))



def classify_records(
    records: pd.DataFrame,
    taxonomy: pd.DataFrame,
    taxonomy_columns: list[str],
    label_examples: pd.DataFrame | None,
    config: dict[str, Any],
    config_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    output = records.copy()
    warnings: list[str] = []
    if output.empty:
        for depth in range(1, 5):
            output[f"l{depth}"] = ""
        output["confidence"] = 0.0
        return output, warnings

    embedder = Embedder(config, config_dir)
    if embedder.error:
        warnings.append(f"Embedding model unavailable; lexical fallback used: {embedder.error[:300]}")

    # Descriptive fields only: IDs, golden keys and target labels are not model features.
    query_texts = [build_semantic_text(row) for _, row in output.iterrows()]

    max_depth = min(4, len(taxonomy_columns))
    taxonomy_rows_by_level: dict[int, list[tuple[str, ...]]] = {}
    taxonomy_texts_by_level: dict[int, list[str]] = {}
    for level in range(max_depth):
        entries: set[tuple[str, ...]] = set()
        for _, tax_row in taxonomy.iterrows():
            values = tuple(str(tax_row.get(column, "")).strip() for column in taxonomy_columns[: level + 1])
            if values[-1] and all(values[:-1]):
                entries.add(values)
        ordered = sorted(entries)
        taxonomy_rows_by_level[level] = ordered
        taxonomy_texts_by_level[level] = [" > ".join(path) for path in ordered]

    query_vectors = embedder.encode(query_texts, query=True) if embedder.available else None
    taxonomy_vectors: dict[int, np.ndarray | None] = {}
    for level in range(max_depth):
        taxonomy_vectors[level] = (
            embedder.encode(taxonomy_texts_by_level[level], query=False)
            if embedder.available else None
        )

    # Aggregate historical examples by every taxonomy prefix. This keeps the
    # scoring loop O(number of candidate branches), not O(all labelled rows).
    prefix_vector_sums: dict[int, dict[tuple[str, ...], np.ndarray]] = {
        level: {} for level in range(max_depth)
    }
    prefix_vector_counts: dict[int, Counter[tuple[str, ...]]] = {
        level: Counter() for level in range(max_depth)
    }
    prefix_token_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
        level: defaultdict(Counter) for level in range(max_depth)
    }
    label_vectors_by_record: dict[str, np.ndarray] = {}
    label_tokens_by_record: dict[str, set[str]] = {}
    label_path_by_record: dict[str, tuple[str, ...]] = {}

    if (
        config.get("use_label_memory", True)
        and label_examples is not None
        and not label_examples.empty
        and "label_text" in label_examples.columns
    ):
        usable = label_examples.loc[label_examples["label_path"].map(bool)].copy()
        label_texts = usable["label_text"].astype(str).tolist()
        label_matrix = embedder.encode(label_texts, query=True) if embedder.available and label_texts else None

        for index, (_, ref_row) in enumerate(usable.iterrows()):
            record_id = str(ref_row["record_id"])
            label_path = tuple(str(value).strip() for value in ref_row["label_path"] if str(value).strip())
            label_tokens = _lexical_tokens(str(ref_row["label_text"]))
            label_path_by_record[record_id] = label_path
            label_tokens_by_record[record_id] = label_tokens

            for level in range(min(max_depth, len(label_path))):
                prefix = label_path[: level + 1]
                prefix_token_counts[level][prefix].update(label_tokens)
                if label_matrix is not None:
                    if prefix not in prefix_vector_sums[level]:
                        prefix_vector_sums[level][prefix] = np.zeros(
                            label_matrix.shape[1], dtype=np.float32
                        )
                    prefix_vector_sums[level][prefix] += label_matrix[index]
                    prefix_vector_counts[level][prefix] += 1

            if label_matrix is not None:
                label_vectors_by_record[record_id] = label_matrix[index]

    label_memory_available = any(
        prefix_token_counts[level] or prefix_vector_sums[level] for level in range(max_depth)
    )
    label_weight = float(config.get("label_memory_weight", 0.0)) if label_memory_available else 0.0
    taxonomy_weight = float(config.get("taxonomy_weight", 1.0))
    lexical_weight = max(0.0, min(1.0, float(config.get("lexical_weight", 0.2))))

    result_paths: list[list[str]] = []
    confidences: list[float] = []
    for row_index, (_, record) in enumerate(output.iterrows()):
        query_text = query_texts[row_index]
        query_vector = query_vectors[row_index] if query_vectors is not None else None
        query_tokens = _lexical_tokens(query_text)
        current_id = str(record["record_id"])
        current_label_path = label_path_by_record.get(current_id, ())
        current_label_vector = label_vectors_by_record.get(current_id)
        current_label_tokens = label_tokens_by_record.get(current_id, set())

        chosen: list[str] = []
        parent: tuple[str, ...] = ()
        level_scores: list[float] = []

        for level in range(max_depth):
            candidates = [
                path for path in taxonomy_rows_by_level[level]
                if path[: len(parent)] == parent
            ]
            if not candidates:
                break

            scores: list[float] = []
            candidate_positions = {
                path: position for position, path in enumerate(taxonomy_rows_by_level[level])
            }
            for candidate in candidates:
                position = candidate_positions[candidate]
                tax_vector = (
                    taxonomy_vectors[level][position]
                    if taxonomy_vectors[level] is not None else None
                )
                semantic_score = _cosine_score(query_vector, tax_vector)
                lexical_score = token_similarity(query_text, " ".join(candidate))
                if embedder.available:
                    taxonomy_score = (1.0 - lexical_weight) * semantic_score + lexical_weight * lexical_score
                else:
                    # In lexical-only mode, lexical similarity must remain the full ranking signal.
                    taxonomy_score = lexical_score

                label_score = 0.0
                if label_weight > 0:
                    if embedder.available:
                        vector_sum = prefix_vector_sums[level].get(candidate)
                        vector_count = int(prefix_vector_counts[level].get(candidate, 0))
                        if (
                            vector_sum is not None
                            and current_label_vector is not None
                            and current_label_path[: level + 1] == candidate
                        ):
                            vector_sum = vector_sum - current_label_vector
                            vector_count -= 1
                        if vector_sum is not None and vector_count > 0:
                            reference_vector = vector_sum / vector_count
                            reference_norm = float(np.linalg.norm(reference_vector))
                            if reference_norm > 1e-12:
                                label_score = _cosine_score(
                                    query_vector, reference_vector / reference_norm
                                )
                    else:
                        reference_counter = prefix_token_counts[level].get(candidate)
                        if reference_counter:
                            adjusted_counter = reference_counter.copy()
                            if current_label_path[: level + 1] == candidate:
                                adjusted_counter.subtract(current_label_tokens)
                            reference_tokens = {
                                token for token, count in adjusted_counter.items() if count > 0
                            }
                            union = query_tokens | reference_tokens
                            label_score = (
                                len(query_tokens & reference_tokens) / len(union)
                                if union else 0.0
                            )

                combined = (
                    taxonomy_weight * taxonomy_score + label_weight * label_score
                ) / max(taxonomy_weight + label_weight, 1e-9)
                scores.append(float(combined))

            best_index = int(np.argmax(scores))
            best_path = candidates[best_index]
            chosen.append(best_path[-1])
            parent = best_path
            level_scores.append(float(scores[best_index]))

        result_paths.append(chosen[:4] + [""] * max(0, 4 - len(chosen)))
        confidence = max(0.0, min(1.0, float(np.mean(level_scores)))) if level_scores else 0.0
        if not query_text.strip():
            confidence = 0.0
        confidences.append(confidence)

    for depth in range(4):
        output[f"l{depth + 1}"] = [path[depth] for path in result_paths]
    output["confidence"] = confidences
    if not embedder.available:
        warnings.append("Classifier ran in lexical-only fallback mode; evaluate quality before blind use")
    return output, warnings
