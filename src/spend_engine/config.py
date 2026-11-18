from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a company configuration is incomplete or invalid."""


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a YAML mapping")
    validate_config(config)
    return config, path.parent


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("company", "files", "sources", "taxonomy", "classification", "output")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {', '.join(missing)}")

    company = config["company"]
    if not company.get("id") or not company.get("reporting_currency"):
        raise ConfigurationError("company.id and company.reporting_currency are required")

    for source_name, source in config["sources"].items():
        for key in ("record_id_column", "amount_column", "amount_type"):
            value = source.get(key)
            if not value or (isinstance(value, str) and value.startswith("REPLACE_")):
                raise ConfigurationError(
                    f"Source '{source_name}' needs a valid '{key}' (template placeholders are not runnable)"
                )
        if source["amount_type"] not in {"net", "gross"}:
            raise ConfigurationError(f"Source '{source_name}': amount_type must be net or gross")
        if source.get("status_required") and not source.get("status_column"):
            raise ConfigurationError(f"Source '{source_name}' requires a status_column")
        description_columns = source.get("description_columns", [])
        text_fields = [column for column in description_columns if column]
        if not text_fields and not source.get("account_description_column"):
            raise ConfigurationError(
                f"Source '{source_name}' needs at least one descriptive text column for classification"
            )
        for field, value in source.items():
            values = value if isinstance(value, list) else [value]
            if any(isinstance(item, str) and item.startswith("REPLACE_") for item in values):
                raise ConfigurationError(
                    f"Source '{source_name}' still contains a template placeholder in '{field}'"
                )

    taxonomy_columns = config["taxonomy"].get("columns", [])
    if not taxonomy_columns or any(str(c).startswith("REPLACE_") for c in taxonomy_columns):
        raise ConfigurationError("taxonomy.columns must contain the actual L1..L4 column names")


def resolve_path(config_dir: Path, value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()
