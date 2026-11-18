from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigurationError
from .io import read_table
from .normalize import text_value


def load_taxonomy(path: Path, columns: list[str], csv_config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[str]]:
    raw = read_table(path, csv_config)
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ConfigurationError(f"Taxonomy is missing configured columns {missing}; found {list(raw.columns)}")

    taxonomy = raw[columns].copy()
    for column in columns:
        taxonomy[column] = taxonomy[column].map(text_value)
    taxonomy = taxonomy.loc[taxonomy[columns[0]] != ""].drop_duplicates().reset_index(drop=True)

    warnings: list[str] = []
    # A tree may omit levels below its leaves, but it may not skip a parent and
    # then define a child. Treat malformed hierarchy as a config/data error.
    malformed_paths: list[str] = []
    for index, row in taxonomy.iterrows():
        parent_is_blank = False
        for column in columns:
            value = row[column]
            if parent_is_blank and value:
                malformed_paths.append(f"row {index + 2}: {column}='{value}' has a blank parent")
            if not value:
                parent_is_blank = True
    if malformed_paths:
        raise ConfigurationError(
            "Taxonomy contains child categories with missing parent levels: "
            + "; ".join(malformed_paths[:10])
        )
    if taxonomy.empty:
        raise ConfigurationError("Taxonomy has no usable categories")
    return taxonomy, warnings


def taxonomy_paths(taxonomy: pd.DataFrame, columns: list[str]) -> list[str]:
    paths = []
    for _, row in taxonomy.iterrows():
        path = [str(row[column]).strip() for column in columns if str(row[column]).strip()]
        paths.append(" > ".join(path))
    return paths
