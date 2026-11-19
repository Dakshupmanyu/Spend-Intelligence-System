"""Label memory built from a *training* file, never from the rows being classified.

Why: if the memory is built from the same rows we then classify, every row can find itself
(or its identical siblings) and accuracy looks far better than it will on new data. A new
batch (A2, B2) has no labels at all, so memory must come from the original labelled file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hashlib

import pandas as pd

from .canonical import canonicalize_source
from .config import resolve_path
from .features import build_semantic_text
from .io import read_table
from .normalize import normalize_text

LABEL_COLUMNS = ("label_l1", "label_l2", "label_l3", "label_l4")
EXAMPLE_COLUMNS = ["record_id", "label_path", "label_text"]


def record_fingerprint(row: Any) -> str:
    """Content fingerprint: tells 'the same record' apart from 'a different record with a reused id'."""
    parts = [
        normalize_text(row.get("description", "")),
        normalize_text(row.get("supplier_name", "")),
        str(row.get("amount_raw", "")).strip(),
    ]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def examples_from_labelled_records(records: pd.DataFrame) -> pd.DataFrame:
    """Turn records that carry label_l1..label_l4 into reference examples (path + text)."""
    if records.empty:
        return pd.DataFrame(columns=EXAMPLE_COLUMNS)
    frame = records.copy()
    frame["label_path"] = [
        tuple(str(row.get(column, "")).strip() for column in LABEL_COLUMNS if str(row.get(column, "")).strip())
        for _, row in frame.iterrows()
    ]
    frame = frame.loc[frame["label_path"].map(bool)].copy()
    if frame.empty:
        return pd.DataFrame(columns=EXAMPLE_COLUMNS)
    frame["label_text"] = [build_semantic_text(row) for _, row in frame.iterrows()]
    frame["fingerprint"] = [record_fingerprint(row) for _, row in frame.iterrows()]
    return frame[EXAMPLE_COLUMNS + ["fingerprint"]].drop_duplicates("record_id")


def build_memory_examples(
    config: dict[str, Any],
    config_dir: Path,
    run_records: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Read each configured training file, join its labels by exact record id, return examples.

    config["labels"]["memory_sources"] is a list of {source: <key under sources>, file: <path>}.
    The training file must have the same layout as the configured source it names.
    Examples whose record id also appears in the current run are dropped (exclude_run_ids,
    default true): a run row must never be classified with its own label.
    """
    warnings: list[str] = []
    label_cfg = config.get("labels", {})
    csv_cfg = config.get("csv", {})
    labels_path = resolve_path(config_dir, config.get("files", {}).get("labels"))
    if labels_path is None or not labels_path.is_file():
        warnings.append("Label memory requested but the labels file is missing; memory is empty")
        return pd.DataFrame(columns=EXAMPLE_COLUMNS), warnings

    labels = read_table(labels_path, csv_cfg)
    join_column = label_cfg.get("join_column", "record_id")
    category_columns = list(label_cfg.get("category_columns", []))[:4]
    missing = [c for c in [join_column, *category_columns] if c not in labels.columns]
    if missing:
        warnings.append(f"Labels file lacks columns {missing}; memory is empty")
        return pd.DataFrame(columns=EXAMPLE_COLUMNS), warnings

    labels = labels.copy()
    labels["_join_key"] = labels[join_column].astype(str).str.strip()
    duplicated = int(labels["_join_key"].duplicated().sum())
    if duplicated:
        warnings.append(f"Labels file has {duplicated} duplicate ids; the first label per id is used")
    labels = labels.drop_duplicates("_join_key", keep="first").set_index("_join_key")
    include_column = label_cfg.get("include_column")

    frames: list[pd.DataFrame] = []
    for entry in label_cfg.get("memory_sources", []):
        source_key = entry["source"]
        path = resolve_path(config_dir, entry["file"])
        if path is None or not path.is_file():
            warnings.append(f"Memory training file for '{source_key}' not found: {entry.get('file')}; skipped")
            continue
        raw = read_table(path, csv_cfg)
        canonical, canon_warnings = canonicalize_source(raw, source_key, config["sources"][source_key], config["company"]["id"])
        warnings.extend(canon_warnings)
        key_column = "record_id" if join_column == "record_id" else "raw_record_id"
        canonical = canonical.loc[canonical[key_column].astype(str).str.strip().isin(labels.index)].copy()
        if canonical.empty:
            warnings.append(f"No rows of '{path.name}' have a label; nothing added to memory")
            continue
        joined = labels.loc[canonical[key_column].astype(str).str.strip()]
        for depth, column in enumerate(category_columns, start=1):
            canonical[f"label_l{depth}"] = joined[column].astype(str).str.strip().values
        if include_column and include_column in joined.columns:
            keep = joined[include_column].astype(str).str.strip().isin({"1", "1.0", "true", "True"}).values
            canonical = canonical.loc[keep]
        frames.append(canonical)

    if not frames:
        return pd.DataFrame(columns=EXAMPLE_COLUMNS), warnings
    examples = examples_from_labelled_records(pd.concat(frames, ignore_index=True))

    if label_cfg.get("exclude_run_ids", True) and not run_records.empty:
        # A run row must never be classified with its own label. Same id AND same content = same
        # record, so drop it from memory. Same id but different content = an id reused by a new
        # record: the old example is still a valid example of the OLD record, so it is kept.
        run_prints = {
            str(record_id): record_fingerprint(row)
            for record_id, (_, row) in zip(run_records["record_id"].astype(str), run_records.iterrows())
        }
        same_id = examples["record_id"].astype(str).isin(run_prints)
        same_content = pd.Series(
            [
                bool(shared) and run_prints.get(str(rid)) == fingerprint
                for rid, fingerprint, shared in zip(examples["record_id"], examples["fingerprint"], same_id)
            ],
            index=examples.index,
        )
        reused = same_id & ~same_content
        if same_content.any():
            warnings.append(
                f"{int(same_content.sum())} memory example(s) are the same record as a row in this run and were "
                "removed (a row must not see its own label)."
            )
        if reused.any():
            warnings.append(
                f"{int(reused.sum())} record id(s) are reused by different records in this run; the old labelled "
                "examples are kept, and no label is ever attached to the new rows."
            )
        examples = examples.loc[~same_content].copy()
    return examples, warnings
