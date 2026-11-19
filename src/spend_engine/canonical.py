from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .io import clean_cell, require_columns
from .normalize import normalize_identifier, text_value


CANONICAL_COLUMNS = [
    "record_id", "raw_record_id", "source", "source_namespace", "amount_raw", "amount_type",
    "vat_raw", "currency", "row_fx_rate_raw", "date_raw", "fallback_date_raw", "fx_date_raw", "date_parsed",
    "date_parse_failed", "supplier_id", "supplier_name", "item_id", "item_namespace", "description",
    "po_number", "po_item", "po_reference", "status", "invoice_number", "invoice_line", "account_code",
    "account_description", "document_type", "quantity", "uom", "source_row_index",
]


def _get(row: pd.Series, column: str | None) -> str:
    return clean_cell(row[column]) if column and column in row.index else ""


def _join_columns(row: pd.Series, columns: list[str] | None) -> str:
    values = [_get(row, column) for column in (columns or [])]
    return " | ".join(value for value in values if value)


def extract_po_and_item(value: Any) -> tuple[str, str]:
    raw = text_value(value)
    if not raw:
        return "", ""
    # PO references commonly look like 4510000018/10.0, PO 4510000018 or PO#4510000018.
    match = re.search(r"(?<!\d)(\d{6,20})(?:\s*[/\\-]\s*(\d{1,6})(?:\.0+)?)?", raw)
    if not match:
        return normalize_identifier(raw), ""
    po = normalize_identifier(match.group(1), strip_leading_zeroes=True)
    item = normalize_identifier(match.group(2), strip_leading_zeroes=True) if match.group(2) else ""
    return po, item


def canonicalize_source(
    raw: pd.DataFrame,
    source_name: str,
    source_config: dict[str, Any],
    company_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    rid_col = source_config["record_id_column"]
    amount_col = source_config["amount_column"]
    required_columns = [rid_col, amount_col]
    if source_config.get("currency_required") and source_config.get("currency_column"):
        required_columns.append(source_config["currency_column"])
    if source_config.get("status_required") and source_config.get("status_column"):
        required_columns.append(source_config["status_column"])
    required_columns.extend(source_config.get("namespace_required_columns", []))
    required_columns.extend(source_config.get("item_namespace_required_columns", []))
    require_columns(raw, list(dict.fromkeys(required_columns)), source_name)
    skipped_lines = int(raw.attrs.get("bad_lines_skipped", 0))
    if skipped_lines:
        warnings.append(f"{source_name}: {skipped_lines} malformed CSV line(s) skipped by the reader")

    # Junk rows (blank record id, or a repeated header row pasted mid-file) are not records.
    # Drop them with a warning instead of aborting the whole run.
    id_values = raw[rid_col].map(clean_cell)
    is_blank = id_values == ""
    is_repeated_header = id_values.str.casefold() == str(rid_col).strip().casefold()
    junk = is_blank | is_repeated_header
    if junk.any():
        warnings.append(
            f"{source_name}: dropped {int(junk.sum())} junk row(s) "
            f"({int(is_blank.sum())} blank '{rid_col}', {int(is_repeated_header.sum())} repeated header); "
            f"first CSV rows {[int(p) + 2 for p in raw.index[junk][:10]]}"
        )
        raw = raw.loc[~junk]

    optional_single_fields = [
        "vat_column", "currency_column", "row_fx_rate_column", "date_column",
        "fallback_fx_date_column", "supplier_id_column", "supplier_name_column", "item_id_column",
        "po_number_column", "po_item_column", "po_reference_column", "status_column",
        "paid_date_column", "invoice_number_column", "invoice_line_column", "account_code_column",
        "account_description_column", "document_type_column", "quantity_column", "uom_column",
    ]
    for field in optional_single_fields:
        col = source_config.get(field)
        if col and col not in raw.columns:
            warnings.append(f"{source_name}: configured optional column '{col}' is absent; treating as blank")
    if not source_config.get("status_column") and source_config.get("status_absent_policy") == "include_with_warning":
        warnings.append(
            f"{source_name}: no approval-status column configured; rows without explicit paid evidence "
            "follow the configured include-with-warning policy"
        )

    description_columns = source_config.get("description_columns", [])
    missing_desc = [col for col in description_columns if col not in raw.columns]
    if missing_desc:
        warnings.append(f"{source_name}: description columns absent {missing_desc}; available text fields will be used")

    rows: list[dict[str, Any]] = []
    prefix = source_config.get("record_id_prefix", "")
    namespace_columns = source_config.get("supplier_namespace_columns", [])
    item_namespace_columns = source_config.get("item_namespace_columns", namespace_columns)
    for index, row in raw.iterrows():
        raw_id = _get(row, rid_col)
        record_id = f"{prefix}{raw_id}"
        po_number = _get(row, source_config.get("po_number_column"))
        po_item = _get(row, source_config.get("po_item_column"))
        po_reference = _get(row, source_config.get("po_reference_column"))
        if po_reference and not po_number:
            po_number, parsed_item = extract_po_and_item(po_reference)
            po_item = po_item or parsed_item
        else:
            po_number = normalize_identifier(po_number, strip_leading_zeroes=True)
            po_item = normalize_identifier(po_item, strip_leading_zeroes=True)
        date_raw = _get(row, source_config.get("date_column"))
        fallback_date = _get(row, source_config.get("fallback_fx_date_column"))
        paid_date = _get(row, source_config.get("paid_date_column"))
        status = _get(row, source_config.get("status_column"))
        if not status and paid_date:
            status = "paid"
        namespace = "|".join(_get(row, col) for col in namespace_columns if col in raw.columns)
        if not namespace:
            namespace = str(source_config.get("source_name", source_name))
        item_namespace = "|".join(_get(row, col) for col in item_namespace_columns if col in raw.columns)
        if not item_namespace:
            item_namespace = str(source_config.get("source_name", source_name))
        rows.append({
            "record_id": record_id,
            "raw_record_id": raw_id,
            "source": str(source_config.get("source_name", source_name)),
            "source_namespace": namespace,
            "amount_raw": _get(row, amount_col),
            "amount_type": source_config.get("amount_type", "net"),
            "vat_raw": _get(row, source_config.get("vat_column")),
            "currency": _get(row, source_config.get("currency_column")).upper(),
            "row_fx_rate_raw": _get(row, source_config.get("row_fx_rate_column")),
            "date_raw": date_raw,
            "fallback_date_raw": fallback_date,
            "fx_date_raw": date_raw or fallback_date,
            "date_parsed": pd.NaT,
            "date_parse_failed": False,
            "supplier_id": _get(row, source_config.get("supplier_id_column")),
            "supplier_name": _get(row, source_config.get("supplier_name_column")),
            "item_id": _get(row, source_config.get("item_id_column")),
            "item_namespace": item_namespace,
            "description": _join_columns(row, description_columns),
            "po_number": po_number,
            "po_item": po_item,
            "po_reference": po_reference,
            "status": status,
            "invoice_number": _get(row, source_config.get("invoice_number_column")),
            "invoice_line": _get(row, source_config.get("invoice_line_column")),
            "account_code": _get(row, source_config.get("account_code_column")),
            "account_description": _get(row, source_config.get("account_description_column")),
            "document_type": _get(row, source_config.get("document_type_column")),
            "quantity": _get(row, source_config.get("quantity_column")),
            "uom": _get(row, source_config.get("uom_column")),
            "source_row_index": int(index),
        })
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS), warnings
