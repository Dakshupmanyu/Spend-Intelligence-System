from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
    "l1", "l2", "l3", "l4", "confidence",
]
VALID_EXCLUSIONS = {"", "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


class OutputValidationError(RuntimeError):
    pass


def validate_output(
    source_records: pd.DataFrame,
    output: pd.DataFrame,
    taxonomy: pd.DataFrame,
    taxonomy_columns: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if list(output.columns) != OUTPUT_COLUMNS:
        errors.append(f"Output columns differ from required schema: {list(output.columns)}")
    if len(output) != len(source_records):
        errors.append(f"Row count mismatch: inputs={len(source_records)}, outputs={len(output)}")
    if output["record_id"].duplicated().any():
        errors.append("Output contains duplicate record_id values")

    input_ids = set(source_records["record_id"].astype(str))
    output_ids = set(output["record_id"].astype(str))
    if input_ids != output_ids:
        errors.append(f"Record ID mismatch: missing={len(input_ids - output_ids)}, extra={len(output_ids - input_ids)}")

    invalid_reasons = sorted(set(output["exclusion_reason"].fillna("").astype(str)) - VALID_EXCLUSIONS)
    if invalid_reasons:
        errors.append(f"Invalid exclusion_reason values: {invalid_reasons}")

    include = pd.to_numeric(output["include"], errors="coerce")
    if include.isna().any() or not set(include.astype(int).unique()).issubset({0, 1}):
        errors.append("include must contain only 0 or 1")
    excluded = include == 0
    spend = pd.to_numeric(output["spend"], errors="coerce")
    if spend.isna().any() or not np.isfinite(spend.to_numpy(dtype=float)).all():
        errors.append("spend contains missing or non-finite values")
    if (spend[excluded].abs() > 1e-9).any():
        errors.append("Excluded rows must have spend=0")
    if (output.loc[excluded, "exclusion_reason"].astype(str).str.strip() == "").any():
        errors.append("Every excluded row must have an exclusion_reason")
    if (output.loc[~excluded, "exclusion_reason"].astype(str).str.strip() != "").any():
        errors.append("Included rows must have a blank exclusion_reason")

    confidence = pd.to_numeric(output["confidence"], errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        errors.append("confidence must be numeric and between 0 and 1")

    valid_paths = set()
    for _, row in taxonomy.iterrows():
        path = tuple(str(row.get(column, "")).strip() for column in taxonomy_columns)
        valid_paths.add(path)
    for _, row in output.loc[~excluded].iterrows():
        path = tuple(str(row.get(f"l{i}", "")).strip() for i in range(1, 5))
        populated = tuple(value for value in path if value)
        if not populated:
            errors.append(f"Included row {row['record_id']} has no category")
            continue
        if not any(tuple(value for value in candidate if value) == populated for candidate in valid_paths):
            errors.append(f"Included row {row['record_id']} has a category path outside the taxonomy: {populated}")

    if errors:
        raise OutputValidationError("Output validation failed:\n- " + "\n- ".join(errors[:30]))

    return {
        "input_rows": int(len(source_records)),
        "output_rows": int(len(output)),
        "included_rows": int((include == 1).sum()),
        "excluded_rows": int((include == 0).sum()),
        "included_spend_total": float(spend[include == 1].sum()),
        "unique_suppliers": int(output.loc[include == 1, "supplier_key"].nunique()),
        "unique_items": int(output.loc[include == 1, "item_key"].nunique()),
    }
