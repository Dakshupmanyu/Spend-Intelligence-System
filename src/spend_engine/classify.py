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



def _token_matrix(token_sets: list[set[str]], vocab: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(token_sets), len(vocab)), dtype=np.float32)
    for row, tokens in enumerate(token_sets):
        for token in tokens:
            column = vocab.get(token)
            if column is not None:
                matrix[row, column] = 1.0
    return matrix


def _jaccard(query_bin: np.ndarray, query_sizes: np.ndarray, node_bin: np.ndarray, node_sizes: np.ndarray) -> np.ndarray:
    """Vectorised Jaccard between every query and every node (same rule as normalize.token_similarity)."""
    inter = query_bin @ node_bin.T
    union = query_sizes[:, None] + node_sizes[None, :] - inter
    result = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    result[query_sizes == 0, :] = 0.0
    result[:, node_sizes == 0] = 0.0
    return result


def _build_levels(taxonomy: pd.DataFrame, taxonomy_columns: list[str], max_depth: int):
    """Unique prefixes per level, plus parent->children position maps. Ragged trees are fine."""
    paths_by_level: list[list[tuple[str, ...]]] = []
    for level in range(max_depth):
        entries: set[tuple[str, ...]] = set()
        for _, tax_row in taxonomy.iterrows():
            values = tuple(str(tax_row.get(column, "")).strip() for column in taxonomy_columns[: level + 1])
            if values[-1] and all(values[:-1]):
                entries.add(values)
        paths_by_level.append(sorted(entries))
    children: list[dict[int, list[int]]] = [dict() for _ in range(max_depth)]
    for level in range(1, max_depth):
        parent_index = {path: pos for pos, path in enumerate(paths_by_level[level - 1])}
        for pos, path in enumerate(paths_by_level[level]):
            parent = parent_index.get(path[:-1])
            if parent is not None:
                children[level].setdefault(parent, []).append(pos)
    return paths_by_level, children


def classify_records(
    records: pd.DataFrame,
    taxonomy: pd.DataFrame,
    taxonomy_columns: list[str],
    label_examples: pd.DataFrame | None,
    config: dict[str, Any],
    config_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Beam-search classification into the taxonomy read at run time (any depth 1..4).

    Every taxonomy node (a prefix such as L1 > L2) is scored against the row text:
      - semantic similarity to the node path text (embeddings, when available)
      - word overlap with the node path text
      - similarity to historical labelled examples under that node (label memory, optional)
    Upper levels are scored with a look-ahead: a node is as good as its best descendant leaf, so
    a vague branch name cannot hide the right branch. A beam keeps several competing paths, so a
    wrong choice at L1 can still be recovered at L2/L3. Rows with no description get the majority
    path of the same item, then the same supplier, with low confidence.
    """
    output = records.copy()
    warnings: list[str] = []
    for depth in range(1, 5):
        output[f"l{depth}"] = ""
    output["confidence"] = 0.0
    if output.empty:
        return output, warnings

    embedder = Embedder(config, config_dir)
    if embedder.error:
        warnings.append(f"Embedding model unavailable; lexical fallback used: {embedder.error[:300]}")

    max_depth = min(4, len(taxonomy_columns))
    paths_by_level, children = _build_levels(taxonomy, taxonomy_columns, max_depth)
    if not paths_by_level or not paths_by_level[0]:
        warnings.append("Taxonomy is empty; no classification was possible")
        return output, warnings

    beam_width = max(1, int(config.get("beam_width", 5)))
    lookahead = max(0.0, min(1.0, float(config.get("lookahead_weight", 0.5))))
    taxonomy_weight = float(config.get("taxonomy_weight", 1.0))
    lexical_weight = max(0.0, min(1.0, float(config.get("lexical_weight", 0.2))))

    # Descriptive fields only: IDs, golden keys and target labels are never model features.
    row_texts = [build_semantic_text(row) for _, row in output.iterrows()]
    unique_texts = sorted(set(row_texts))
    text_index = {text: i for i, text in enumerate(unique_texts)}
    query_tokens = [_lexical_tokens(text) for text in unique_texts]
    query_sizes = np.array([len(t) for t in query_tokens], dtype=np.float32)
    query_vectors = embedder.encode(unique_texts, query=True) if embedder.available else None

    node_texts = [[" ".join(path) for path in level] for level in paths_by_level]
    node_tokens = [[_lexical_tokens(t) for t in level] for level in node_texts]
    node_vectors: list[np.ndarray | None] = [
        embedder.encode([" > ".join(path) for path in level], query=False) if embedder.available else None
        for level in paths_by_level
    ]

    # ---- label memory: per-node reference vectors / token sets built from historical examples ----
    label_weight = 0.0
    ref_vectors: list[np.ndarray | None] = [None] * max_depth
    ref_tokens: list[list[set[str]] | None] = [None] * max_depth
    if (
        config.get("use_label_memory", True)
        and label_examples is not None
        and not label_examples.empty
        and "label_text" in label_examples.columns
    ):
        usable = label_examples.loc[label_examples["label_path"].map(bool)]
        if not usable.empty:
            label_texts = usable["label_text"].astype(str).tolist()
            label_paths = [tuple(str(v).strip() for v in path if str(v).strip()) for path in usable["label_path"]]
            unique_label_texts = sorted(set(label_texts))
            label_unique_index = {t: i for i, t in enumerate(unique_label_texts)}
            label_matrix = embedder.encode(unique_label_texts, query=True) if embedder.available else None
            for level in range(max_depth):
                position = {path: i for i, path in enumerate(paths_by_level[level])}
                tokens_per_node: list[set[str]] = [set() for _ in paths_by_level[level]]
                sums = None
                counts = np.zeros(len(paths_by_level[level]), dtype=np.float32)
                if label_matrix is not None:
                    sums = np.zeros((len(paths_by_level[level]), label_matrix.shape[1]), dtype=np.float32)
                for text, path in zip(label_texts, label_paths):
                    node = position.get(path[: level + 1]) if len(path) >= level + 1 else None
                    if node is None:
                        continue
                    tokens_per_node[node] |= _lexical_tokens(text)
                    counts[node] += 1
                    if sums is not None:
                        sums[node] += label_matrix[label_unique_index[text]]
                if counts.any():
                    label_weight = float(config.get("label_memory_weight", 0.0))
                    if sums is not None:
                        norms = np.linalg.norm(sums, axis=1, keepdims=True)
                        ref_vectors[level] = np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 1e-12)
                    ref_tokens[level] = tokens_per_node

    vocab_tokens: set[str] = set()
    for level_tokens in node_tokens:
        for tokens in level_tokens:
            vocab_tokens |= tokens
    for level in range(max_depth):
        for tokens in ref_tokens[level] or []:
            vocab_tokens |= tokens
    vocab = {token: i for i, token in enumerate(sorted(vocab_tokens))}
    node_bins = [_token_matrix(level_tokens, vocab) for level_tokens in node_tokens]
    node_sizes = [np.array([len(t) for t in level_tokens], dtype=np.float32) for level_tokens in node_tokens]
    ref_bins = [
        _token_matrix(ref_tokens[level], vocab) if ref_tokens[level] is not None else None for level in range(max_depth)
    ]
    ref_sizes = [
        np.array([len(t) for t in ref_tokens[level]], dtype=np.float32) if ref_tokens[level] is not None else None
        for level in range(max_depth)
    ]

    def own_scores(chunk: np.ndarray) -> list[np.ndarray]:
        """Combined score of every node (per level) for a chunk of unique texts."""
        q_bin = _token_matrix([query_tokens[i] for i in chunk], vocab)
        q_sizes = query_sizes[chunk]
        levels: list[np.ndarray] = []
        for level in range(max_depth):
            lexical = _jaccard(q_bin, q_sizes, node_bins[level], node_sizes[level])
            if embedder.available and query_vectors is not None and node_vectors[level] is not None:
                semantic = query_vectors[chunk] @ node_vectors[level].T
                taxonomy_score = (1.0 - lexical_weight) * semantic + lexical_weight * lexical
            else:
                taxonomy_score = lexical  # lexical-only mode keeps lexical similarity as the full signal
            if label_weight > 0 and ref_bins[level] is not None:
                if embedder.available and query_vectors is not None and ref_vectors[level] is not None:
                    label_score = query_vectors[chunk] @ ref_vectors[level].T
                else:
                    label_score = _jaccard(q_bin, q_sizes, ref_bins[level], ref_sizes[level])
                combined = (taxonomy_weight * taxonomy_score + label_weight * label_score) / max(
                    taxonomy_weight + label_weight, 1e-9
                )
            else:
                combined = taxonomy_score
            levels.append(combined.astype(np.float32))
        return levels

    def with_lookahead(own: list[np.ndarray]) -> list[np.ndarray]:
        """Blend each non-leaf node with its best descendant leaf (bottom-up)."""
        if lookahead <= 0 or max_depth == 1:
            return own
        best_leaf: list[np.ndarray] = [None] * max_depth  # type: ignore[list-item]
        effective: list[np.ndarray] = [None] * max_depth  # type: ignore[list-item]
        best_leaf[-1] = own[-1]
        effective[-1] = own[-1]
        for level in range(max_depth - 2, -1, -1):
            bl = own[level].copy()  # a node without children is itself the leaf
            for parent, kids in children[level + 1].items():
                bl[:, parent] = best_leaf[level + 1][:, kids].max(axis=1)
            best_leaf[level] = bl
            has_kids = np.zeros(own[level].shape[1], dtype=bool)
            has_kids[list(children[level + 1].keys())] = True
            effective[level] = np.where(has_kids[None, :], (1.0 - lookahead) * own[level] + lookahead * bl, own[level])
        return effective

    def beam_paths(own_row: list[np.ndarray], eff_row: list[np.ndarray]) -> tuple[list[int], float]:
        beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
        for level in range(max_depth):
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for total, positions in beams:
                if len(positions) < level:  # path already ended (ragged tree)
                    expanded.append((total, positions))
                    continue
                kids = range(len(paths_by_level[0])) if level == 0 else children[level].get(positions[-1], ())
                if not kids:
                    expanded.append((total, positions))
                    continue
                for kid in kids:
                    expanded.append((total + float(eff_row[level][kid]), positions + (kid,)))
            expanded.sort(key=lambda item: (-(item[0] / max(len(item[1]), 1)), item[1]))
            beams = expanded[:beam_width]
        best_total, best_positions = beams[0]
        confidence = float(np.mean([own_row[level][pos] for level, pos in enumerate(best_positions)])) if best_positions else 0.0
        return list(best_positions), confidence

    # ---- score unique texts in chunks, run the beam, remember the best path per unique text ----
    best_path_by_text: list[tuple[str, ...]] = [()] * len(unique_texts)
    confidence_by_text = np.zeros(len(unique_texts), dtype=float)
    chunk_size = 1500
    for start in range(0, len(unique_texts), chunk_size):
        chunk = np.arange(start, min(start + chunk_size, len(unique_texts)))
        own = own_scores(chunk)
        eff = with_lookahead(own)
        for offset, text_id in enumerate(chunk):
            positions, confidence = beam_paths([o[offset] for o in own], [e[offset] for e in eff])
            best_path_by_text[text_id] = tuple(paths_by_level[level][pos][-1] for level, pos in enumerate(positions))
            confidence_by_text[text_id] = max(0.0, min(1.0, confidence))

    paths = [best_path_by_text[text_index[text]] for text in row_texts]
    confidences = [float(confidence_by_text[text_index[text]]) for text in row_texts]
    blank = [not text.strip() for text in row_texts]
    for i, is_blank in enumerate(blank):
        if is_blank:
            confidences[i] = 0.0

    # ---- rows with no text: borrow the majority path of the same item, then the same supplier ----
    if config.get("blank_text_prior", True) and any(blank):
        filled = 0
        for key_column in ("item_key", "supplier_key"):
            if key_column not in output.columns:
                continue
            votes: dict[str, Counter] = defaultdict(Counter)
            keys = output[key_column].astype(str).tolist()
            for key, path, is_blank in zip(keys, paths, blank):
                if not is_blank and path:
                    votes[key][path] += 1
            for i, (key, is_blank) in enumerate(zip(keys, blank)):
                if is_blank and confidences[i] == 0.0 and votes.get(key):
                    path, count = votes[key].most_common(1)[0]
                    paths[i] = path
                    confidences[i] = round(0.2 * count / sum(votes[key].values()), 4)
                    filled += 1
        if filled:
            warnings.append(f"{filled} row(s) had no description and took the majority category of their item/supplier")

    for depth in range(4):
        output[f"l{depth + 1}"] = [path[depth] if depth < len(path) else "" for path in paths]
    output["confidence"] = confidences
    if not embedder.available:
        warnings.append("Classifier ran in lexical-only fallback mode; evaluate quality before blind use")
    return output, warnings
