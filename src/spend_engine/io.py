from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigurationError


def detect_delimiter(path: Path, encoding: str = "utf-8-sig") -> str:
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        sample = handle.read(32768)
    scored: list[tuple[float, str]] = []
    for delimiter in [",", ";", "\t", "|"]:
        try:
            rows = list(csv.reader(sample.splitlines()[:40], delimiter=delimiter))
        except csv.Error:
            continue
        widths = [len(row) for row in rows if row]
        if len(widths) < 2:
            continue
        # Prefer a consistently multi-column parse; this resists semicolons inside comma CSV text.
        from collections import Counter
        mode_width, mode_count = Counter(widths).most_common(1)[0]
        if mode_width > 1:
            consistency = mode_count / len(widths)
            scored.append((mode_width * consistency, delimiter))
    if not scored:
        return ","
    return max(scored, key=lambda item: item[0])[1]


def read_table(path: Path, csv_config: dict[str, Any] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    options = csv_config or {}
    encoding = options.get("encoding", "utf-8-sig")
    delimiter = options.get("delimiter", "auto")
    sep = detect_delimiter(path, encoding) if delimiter == "auto" else delimiter
    on_bad_lines = str(options.get("on_bad_lines", "skip")).lower()

    # Default is "skip": one malformed physical line must never kill a blind-test run.
    # Skipped lines are counted and surfaced as a warning via df.attrs["bad_lines_skipped"].
    # "error" keeps the strict pandas behaviour for debugging.
    skipped: list[int] = []

    def _skip_bad_line(fields: list[str]) -> None:
        skipped.append(len(fields))
        return None

    read_kwargs: dict[str, Any] = {
        "sep": sep,
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
    }
    if on_bad_lines == "error":
        read_kwargs.update(on_bad_lines="error", low_memory=False)
    else:
        # A callable handler is only supported by the python parser.
        read_kwargs.update(on_bad_lines=_skip_bad_line, engine="python")

    # Encoding fallback chain: configured encoding first, then common Windows/Latin exports.
    encodings = [encoding] + [e for e in ("utf-8-sig", "cp1252", "latin-1") if e != encoding]
    last_error: Exception | None = None
    for candidate in encodings:
        skipped.clear()
        try:
            frame = pd.read_csv(path, encoding=candidate, **read_kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise ValueError(f"Could not read '{path}': {exc}") from exc
        frame.attrs["bad_lines_skipped"] = len(skipped)
        frame.attrs["encoding_used"] = candidate
        return frame
    raise ValueError(f"Could not decode '{path}' with {encodings}: {last_error}")


def require_columns(df: pd.DataFrame, columns: list[str], source_name: str) -> None:
    missing = [column for column in columns if column and column not in df.columns]
    if missing:
        raise ConfigurationError(
            f"Source '{source_name}' is missing required configured columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text
