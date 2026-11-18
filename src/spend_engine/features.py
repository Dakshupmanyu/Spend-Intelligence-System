from __future__ import annotations

from typing import Any


SEMANTIC_TEXT_FIELDS = (
    ("description", "description"),
    ("account_description", "account"),
    ("supplier_name", "supplier"),
)


def build_semantic_text(row: Any) -> str:
    """Build a consistent inference/reference string without IDs or target labels."""
    parts: list[str] = []
    for field, label in SEMANTIC_TEXT_FIELDS:
        value = str(row.get(field, "")).strip()
        if value and value.casefold() not in {"nan", "none", "null", "<na>"}:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)
