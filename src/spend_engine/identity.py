from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .normalize import (
    normalize_identifier,
    normalize_supplier_name,
    normalize_text,
    similarity_ratio,
    stable_key,
)


def _mutual_fuzzy_clusters(
    names: list[str], threshold: float, require_same_technical_tokens: bool = False
) -> dict[str, str]:
    """Cluster only mutual-nearest, high-similarity names; avoid broad guesses."""
    unique_names = sorted(set(name for name in names if name))
    if len(unique_names) < 2:
        return {name: name for name in unique_names}

    nearest: dict[str, tuple[str, float] | None] = {}
    for name in unique_names:
        candidates = [
            (similarity_ratio(name, other), other)
            for other in unique_names
            if other != name
            and (
                not require_same_technical_tokens
                or _technical_tokens(name) == _technical_tokens(other)
            )
        ]
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not candidates or candidates[0][0] < threshold:
            nearest[name] = None
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 3:
            nearest[name] = None
            continue
        nearest[name] = (candidates[0][1], candidates[0][0])

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


def _fuzzy_identity_key(
    name: str,
    known_names: dict[str, set[str]],
    threshold: float,
    technical_signature: set[str] | None = None,
    item_mode: bool = False,
) -> str | None:
    candidates: list[tuple[float, str]] = []
    for known_name, keys in known_names.items():
        if len(keys) != 1:
            continue
        if item_mode and technical_signature is not None:
            if _technical_tokens(known_name) != technical_signature:
                continue
        score = similarity_ratio(name, known_name)
        if score >= threshold:
            candidates.append((score, next(iter(keys))))
    candidates.sort(reverse=True)
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 4:
        return None
    return candidates[0][1]


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
        fuzzy_id_key = _fuzzy_identity_key(name, supplier_name_to_keys, threshold=94.0)
        if fuzzy_id_key:
            supplier_keys[index] = fuzzy_id_key
            continue
        canonical_name = supplier_name_clusters.get(name, name)
        supplier_keys[index] = stable_key("SUP", "NAME", canonical_name)

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

        fuzzy_item_key = _fuzzy_identity_key(
            description,
            item_description_to_keys,
            threshold=98.0,
            technical_signature=_technical_tokens(description),
            item_mode=True,
        )
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
