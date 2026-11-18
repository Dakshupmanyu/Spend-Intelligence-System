from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .features import build_semantic_text
from .io import read_table
from .normalize import normalize_identifier, normalize_text


def attach_labels(
    records: pd.DataFrame,
    labels_path: Path | None,
    label_config: dict[str, Any],
    csv_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    output = records.copy()
    warnings: list[str] = []
    for column in ("label_l1", "label_l2", "label_l3", "label_l4", "label_supplier_key", "label_item_key"):
        output[column] = ""
    if not label_config.get("enabled", False):
        return output, pd.DataFrame(), warnings
    if labels_path is None or not labels_path.exists():
        warnings.append("Labels are enabled in config but the configured labels file is unavailable; label memory is empty")
        return output, pd.DataFrame(), warnings

    labels = read_table(labels_path, csv_config)
    join_column = label_config.get("join_column", "record_id")
    if join_column not in labels.columns:
        warnings.append(f"Labels file has no join column '{join_column}'; label memory disabled")
        return output, pd.DataFrame(), warnings

    category_columns = label_config.get("category_columns", [])
    for index, label_col in enumerate(category_columns[:4], start=1):
        if label_col not in labels.columns:
            warnings.append(f"Labels file missing category column '{label_col}'")
        else:
            labels[f"_label_l{index}"] = labels[label_col].astype(str).str.strip()

    labels["_join_key"] = labels[join_column].astype(str).str.strip()
    # Exact identifiers only: no fuzzy join is permitted for gold labels.
    label_map = labels.drop_duplicates("_join_key", keep="first").set_index("_join_key")
    if join_column == "record_id":
        lookup_values = output["record_id"].astype(str)
    else:
        lookup_values = output["raw_record_id"].astype(str)

    for position, record_index in enumerate(output.index):
        key = lookup_values.loc[record_index]
        if key not in label_map.index:
            continue
        label_row = label_map.loc[key]
        if isinstance(label_row, pd.DataFrame):
            label_row = label_row.iloc[0]
        for depth in range(1, 5):
            value = label_row.get(f"_label_l{depth}", "")
            output.at[record_index, f"label_l{depth}"] = str(value).strip() if pd.notna(value) else ""
        supplier_col = label_config.get("supplier_key_column")
        item_col = label_config.get("item_key_column")
        if supplier_col and supplier_col in label_row.index:
            output.at[record_index, "label_supplier_key"] = str(label_row[supplier_col]).strip()
        if item_col and item_col in label_row.index:
            output.at[record_index, "label_item_key"] = str(label_row[item_col]).strip()

    # Create a separate reference table; output rows never copy their own target labels.
    examples = output.copy()
    label_include_column = label_config.get("include_column")
    if label_include_column and label_include_column in labels.columns:
        include_map = labels.drop_duplicates("_join_key").set_index("_join_key")[label_include_column].to_dict()
        examples["_label_include"] = lookup_values.map(lambda key: str(include_map.get(key, "1")).strip())
        examples = examples.loc[examples["_label_include"].isin({"1", "1.0", "true", "True"})]
    examples["label_path"] = examples.apply(
        lambda row: tuple(str(row.get(f"label_l{i}", "")).strip() for i in range(1, 5)
                          if str(row.get(f"label_l{i}", "")).strip()), axis=1
    )
    examples = examples.loc[examples["label_path"].map(bool)].copy()
    examples["label_text"] = examples.apply(build_semantic_text, axis=1)
    examples = examples[["record_id", "label_path", "label_text"]].drop_duplicates("record_id")
    return output, examples, warnings


def label_identity_overrides(records: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], list[str]]:
    supplier: dict[tuple[str, str], str] = {}
    item: dict[tuple[str, str], str] = {}
    supplier_conflicts: set[tuple[str, str]] = set()
    item_conflicts: set[tuple[str, str]] = set()
    warnings: list[str] = []
    for _, row in records.iterrows():
        namespace = str(row.get("source_namespace", ""))
        supplier_id = normalize_identifier(row.get("supplier_id", ""))
        item_id = normalize_identifier(row.get("item_id", ""))
        supplier_key = str(row.get("label_supplier_key", "")).strip()
        item_key = str(row.get("label_item_key", "")).strip()
        if supplier_id and supplier_key:
            key = (namespace, supplier_id)
            if key in supplier_conflicts:
                continue
            if key in supplier and supplier[key] != supplier_key:
                warnings.append(f"Conflicting golden supplier keys for {key}; automatic override skipped")
                supplier.pop(key, None)
                supplier_conflicts.add(key)
            else:
                supplier[key] = supplier_key
        if item_id and item_key:
            key = (str(row.get("item_namespace", row.get("source", ""))), item_id)
            if key in item_conflicts:
                continue
            if key in item and item[key] != item_key:
                warnings.append(f"Conflicting golden item keys for {key}; automatic override skipped")
                item.pop(key, None)
                item_conflicts.add(key)
            else:
                item[key] = item_key
    return supplier, item, warnings
