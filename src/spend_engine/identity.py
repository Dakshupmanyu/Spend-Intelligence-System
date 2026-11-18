from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .normalize import (
    normalize_identifier,
    normalize_supplier_name,
    normalize_text,
    similarity_ratio,
    stable_key,
)


try:  # rapidfuzz does the heavy lifting in C; without it we degrade to a slow pure-python path
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz import process as _process
except ImportError:  # pragma: no cover - rapidfuzz is a core dependency
    _fuzz = None
    _process = None

# Blocks larger than this are not compared all-pairs; they use a sorted-neighbourhood window instead.
BLOCK_CAP = 150
SORTED_WINDOW = 40


def _scores(left: list[str], right: list[str]) -> np.ndarray:
    """Pairwise similarity matrix (0..100) between two name lists."""
    if _process is not None:
        return _process.cdist(left, right, scorer=_fuzz.ratio, dtype=np.float32)
    return np.array([[similarity_ratio(a, b) for b in right] for a in left], dtype=np.float32)


def _block_keys(name: str, signature: str = "") -> set[str]:
    """Cheap keys: two names can only be compared if they share one. Keeps work near-linear."""
    # Prefix, suffix and whole tokens: one typo can break at most one of these for names that
    # are similar enough to merge, so true neighbours still share a block.
    keys = {f"{signature}|p:{name[:6]}", f"{signature}|s:{name[-6:]}"}
    for token in name.split():
        if len(token) >= 3:
            keys.add(f"{signature}|t:{token}")
            if len(token) >= 6:
                keys.add(f"{signature}|h:{token[:3]}|{token[-3:]}")
    return keys


def _candidate_pair_scores(
    names: list[str], min_score: float, signatures: list[str] | None = None
) -> dict[tuple[int, int], float]:
    """Scores for blocked name pairs (i < j) at or above min_score. Never builds an n x n matrix."""
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        for key in _block_keys(name, signatures[index] if signatures else ""):
            blocks[key].append(index)

    pair_scores: dict[tuple[int, int], float] = {}
    seen_groups: set[tuple[int, ...]] = set()
    for members in blocks.values():
        if len(members) < 2:
            continue
        group = tuple(members)
        if group in seen_groups:
            continue
        seen_groups.add(group)
        if len(members) <= BLOCK_CAP:
            matrix = _scores([names[i] for i in members], [names[i] for i in members])
            rows, cols = np.nonzero(np.triu(matrix >= min_score, k=1))
            for r, c in zip(rows.tolist(), cols.tolist()):
                a, b = members[r], members[c]
                pair_scores[(min(a, b), max(a, b))] = float(matrix[r, c])
        else:
            # Huge generic block: compare each name with its sorted neighbours only.
            ordered = sorted(members, key=lambda i: names[i])
            for offset in range(1, SORTED_WINDOW + 1):
                left = ordered[:-offset]
                right = ordered[offset:]
                for a, b in zip(left, right):
                    key = (min(a, b), max(a, b))
                    if key in pair_scores:
                        continue
                    score = _scores([names[a]], [names[b]])[0, 0]
                    if score >= min_score:
                        pair_scores[key] = float(score)
    return pair_scores


def _mutual_fuzzy_clusters(
    names: list[str], threshold: float, require_same_technical_tokens: bool = False
) -> dict[str, str]:
    """Cluster only mutual-nearest, high-similarity names; avoid broad guesses.

    Same rules as before (top match >= threshold, at least 3 points clear of the runner-up,
    and mutual), but candidates come from blocking instead of comparing every pair, so cost
    grows roughly linearly with the number of unique names.
    """
    unique_names = sorted(set(name for name in names if name))
    if len(unique_names) < 2:
        return {name: name for name in unique_names}

    signatures = (
        ["|".join(sorted(_technical_tokens(name))) for name in unique_names]
        if require_same_technical_tokens else None
    )
    # A runner-up only matters when it is within 3 points of the best (>= threshold), so any pair
    # below threshold - 3 can never change a decision and is not stored.
    pair_scores = _candidate_pair_scores(unique_names, threshold - 3.0, signatures)
    neighbours: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (a, b), score in pair_scores.items():
        neighbours[a].append((score, b))
        neighbours[b].append((score, a))

    nearest: dict[str, tuple[str, float] | None] = {}
    for index, name in enumerate(unique_names):
        # Sort by score then name, descending, exactly like the original ordering.
        ranked = sorted(((score, unique_names[j]) for score, j in neighbours.get(index, [])), reverse=True)
        if not ranked or ranked[0][0] < threshold:
            nearest[name] = None
            continue
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 3:
            nearest[name] = None
            continue
        nearest[name] = (ranked[0][1], ranked[0][0])

    parent = {name: name for name in unique_names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for name, choice in nearest.items():
        if choice is None:
            continue
        other, score = choice
        reverse = nearest.get(other)
        if reverse is None or reverse[0] != name or abs(reverse[1] - score) > 1e-9:
            continue
        left_root, right_root = find(name), find(other)
        canonical = min(left_root, right_root)
        parent[left_root] = canonical
        parent[right_root] = canonical

    return {name: find(name) for name in unique_names}


def _technical_tokens(description: str) -> set[str]:
    return {token for token in normalize_text(description).split() if any(char.isdigit() for char in token)}


def _informative_item_description(description: str) -> bool:
    tokens = normalize_text(description).split()
    generic = {
        "item", "items", "product", "products", "part", "parts", "material", "materials",
        "goods", "service", "services", "unknown", "other", "misc", "miscellaneous",
    }
    useful = [token for token in tokens if token not in generic]
    return len(useful) >= 2 or any(any(char.isdigit() for char in token) for token in useful)


class _FuzzyIndex:
    """Blocked nearest-name lookup against names that map to exactly one identity key."""

    def __init__(self, known_names: dict[str, set[str]], item_mode: bool = False):
        self.item_mode = item_mode
        self.names = [name for name, keys in known_names.items() if len(keys) == 1]
        self.key_of = {name: next(iter(known_names[name])) for name in self.names}
        self.signature = {name: "|".join(sorted(_technical_tokens(name))) for name in self.names} if item_mode else {}
        self.blocks: dict[str, list[str]] = defaultdict(list)
        for name in self.names:
            for key in _block_keys(name, self.signature.get(name, "")):
                self.blocks[key].append(name)
        self._cache: dict[tuple[str, float], str | None] = {}

    def lookup(self, name: str, threshold: float, technical_signature: set[str] | None = None) -> str | None:
        cache_key = (name, threshold)
        if cache_key in self._cache:
            return self._cache[cache_key]
        signature = "|".join(sorted(technical_signature)) if (self.item_mode and technical_signature is not None) else ""
        candidates: set[str] = set()
        for key in _block_keys(name, signature):
            members = self.blocks.get(key, [])
            if len(members) <= BLOCK_CAP * 4:  # skip hopelessly generic blocks
                candidates.update(members)
        result: str | None = None
        if candidates:
            ordered = sorted(candidates)
            matrix = _scores([name], ordered)[0]
            ranked = sorted(
                ((float(score), self.key_of[other]) for score, other in zip(matrix, ordered) if score >= threshold),
                reverse=True,
            )
            if ranked and not (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 4):
                result = ranked[0][1]
        self._cache[cache_key] = result
        return result


def assign_entity_keys(records: pd.DataFrame, missing_policy: str = "unique_per_record") -> pd.DataFrame:
    """Resolve supplier/item identities conservatively and deterministically.

    Exact scoped IDs win. Name-only suppliers and description-only items can
    share keys after conservative normalization/fuzzy matching. Ambiguous or
    low-information values are deliberately kept separate.
    """
    output = records.copy()
    supplier_keys: list[str] = []
    item_keys: list[str] = []
    supplier_name_to_keys: dict[str, set[str]] = defaultdict(set)
    item_description_to_keys: dict[str, set[str]] = defaultdict(set)
    idless_supplier_names: list[str] = []
    idless_item_descriptions: list[str] = []
    idless_item_informative: list[bool] = []

    # First pass: create ID-backed keys and deterministic provisional name keys.
    for _, row in output.iterrows():
        supplier_id = normalize_identifier(row.get("supplier_id", ""))
        supplier_name = normalize_supplier_name(row.get("supplier_name", ""))
        supplier_namespace = str(row.get("source_namespace", "")) or str(row.get("source", ""))
        if supplier_id:
            supplier_key = stable_key("SUP", supplier_namespace, supplier_id)
            if supplier_name:
                supplier_name_to_keys[supplier_name].add(supplier_key)
        elif supplier_name:
            idless_supplier_names.append(supplier_name)
            supplier_key = stable_key("SUP", "NAME", supplier_name)
        else:
            identity = str(row["record_id"]) if missing_policy == "unique_per_record" else "UNKNOWN"
            supplier_key = stable_key("SUP", "UNKNOWN", identity)
        supplier_keys.append(supplier_key)

        item_id = normalize_identifier(row.get("item_id", ""))
        item_description = normalize_text(row.get("description", ""))
        informative = _informative_item_description(item_description)
        if item_id:
            item_key = stable_key("ITEM", str(row.get("item_namespace", row.get("source", ""))), item_id)
            if informative:
                item_description_to_keys[item_description].add(item_key)
        elif informative:
            idless_item_descriptions.append(item_description)
            idless_item_informative.append(True)
            item_key = stable_key("ITEM", "DESCRIPTION", item_description)
        elif item_description:
            idless_item_descriptions.append(item_description)
            idless_item_informative.append(False)
            identity = str(row["record_id"]) if missing_policy == "unique_per_record" else "LOW_INFORMATION"
            item_key = stable_key("ITEM", "LOW_INFORMATION", identity)
        else:
            idless_item_descriptions.append("")
            idless_item_informative.append(False)
            identity = str(row["record_id"]) if missing_policy == "unique_per_record" else "UNKNOWN"
            item_key = stable_key("ITEM", "UNKNOWN", identity)
        item_keys.append(item_key)

    supplier_index = _FuzzyIndex(supplier_name_to_keys)
    # Group id-less supplier names only when the best match is mutual and strong.
    supplier_name_clusters = _mutual_fuzzy_clusters(idless_supplier_names, threshold=97.0)
    for index, (_, row) in enumerate(output.iterrows()):
        if normalize_identifier(row.get("supplier_id", "")):
            continue
        name = normalize_supplier_name(row.get("supplier_name", ""))
        if not name:
            continue
        exact_id_keys = supplier_name_to_keys.get(name, set())
        if exact_id_keys:
            if len(exact_id_keys) == 1:
                supplier_keys[index] = next(iter(exact_id_keys))
            # If the same typed name belongs to multiple scoped vendor IDs,
            # keep the name-only record separate instead of guessing an ID.
            continue
        fuzzy_id_key = supplier_index.lookup(name, 94.0)
        if fuzzy_id_key:
            supplier_keys[index] = fuzzy_id_key
            continue
        canonical_name = supplier_name_clusters.get(name, name)
        supplier_keys[index] = stable_key("SUP", "NAME", canonical_name)

    item_index = _FuzzyIndex(item_description_to_keys, item_mode=True)
    # Map id-less item descriptions to an unambiguous ID-backed identity first.
    # Then cluster only near-identical descriptions with the same technical tokens.
    idless_item_name_clusters = _mutual_fuzzy_clusters(
        [name for name, informative in zip(idless_item_descriptions, idless_item_informative) if informative],
        threshold=98.0,
        require_same_technical_tokens=True,
    )
    for index, (_, row) in enumerate(output.iterrows()):
        if normalize_identifier(row.get("item_id", "")):
            continue
        description = normalize_text(row.get("description", ""))
        informative = _informative_item_description(description)
        if not informative:
            continue

        exact_item_keys = item_description_to_keys.get(description, set())
        if len(exact_item_keys) == 1:
            item_keys[index] = next(iter(exact_item_keys))
            continue

        fuzzy_item_key = item_index.lookup(description, 98.0, _technical_tokens(description))
        if fuzzy_item_key:
            item_keys[index] = fuzzy_item_key
            continue

        canonical_description = idless_item_name_clusters.get(description, description)
        item_keys[index] = stable_key("ITEM", "DESCRIPTION", canonical_description)

    output["supplier_key"] = supplier_keys
    output["item_key"] = item_keys
    return output


def inherit_matched_identity(records: pd.DataFrame, matches: dict[str, str]) -> pd.DataFrame:
    """Copy canonical ERP supplier/item keys onto matched invoice-tool records."""
    output = records.copy().set_index("record_id", drop=False)
    for tool_id, erp_id in matches.items():
        if tool_id in output.index and erp_id in output.index:
            output.at[tool_id, "supplier_key"] = output.at[erp_id, "supplier_key"]
            output.at[tool_id, "item_key"] = output.at[erp_id, "item_key"]
    return output.reset_index(drop=True)
