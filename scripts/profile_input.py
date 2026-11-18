from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROLE_HINTS = {
    "record_id": ["line_id", "record_id", "invoice_line_id", "transaction_id", "id"],
    "amount": ["amount", "spend", "gross_amount", "net_amount", "line_amount", "value", "value_zar"],
    "date": ["invoice_date", "posting_date", "document_date", "date", "created_at"],
    "currency": ["currency", "document_currency", "transaction_currency"],
    "supplier_id": ["vendor_id", "supplier_id", "supplier_number"],
    "supplier_name": ["supplier_name", "vendor_name", "payee", "supplier"],
    "item_id": ["material_number", "item_id", "product_code", "sku"],
    "description": ["line_description", "line_text", "description", "material_short_text", "item_text"],
    "status": ["status", "approval_status", "invoice_status", "document_status"],
    "account_code": ["gl_account", "account_code", "ledger_code"],
    "account_description": ["gl_account_description", "account_description", "account_name"],
    "vat_rate": ["vat_rate", "tax_rate", "vat", "tax_percent"],
    "fx_rate": ["currency_rate", "fx_rate", "exchange_rate", "rate_to_usd"],
    "po_number": ["po_number", "purchase_order", "po_reference"],
    "po_item": ["po_item", "po_line", "purchase_order_line"],
    "invoice_number": ["invoice_number", "invoice_no", "document_number"],
    "document_type": ["document_type", "doc_type", "invoice_type"],
    "quantity": ["quantity", "qty", "invoiced_quantity"],
    "uom": ["uom", "unit_of_measure", "unit"],
    "paid_date": ["paid_date", "payment_date", "cleared_date"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a new company's CSV to help draft its YAML config")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/company_profile.json"))
    parser.add_argument("--rows", type=int, default=500, help="Maximum sample rows to inspect")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"File not found: {args.input}")

    sample = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig", nrows=args.rows, sep=None, engine="python")
    columns = []
    for column in sample.columns:
        series = sample[column].astype(str).str.strip()
        nonempty = series[series != ""]
        columns.append({
            "name": column,
            "non_empty_sample": int(len(nonempty)),
            "unique_sample": int(nonempty.nunique()),
            "examples": nonempty.head(5).tolist(),
        })
    normalized_to_original = {re.sub(r"[^a-z0-9]+", "_", col.casefold()).strip("_"): col for col in sample.columns}
    suggestions = {}
    for role, hints in ROLE_HINTS.items():
        candidates = []
        for hint in hints:
            for normalized, original in normalized_to_original.items():
                if hint == normalized:
                    candidates.append({"column": original, "match": "exact"})
                elif hint in normalized or normalized in hint:
                    candidates.append({"column": original, "match": "name_contains"})
        suggestions[role] = candidates[:5]

    result = {
        "file": str(args.input.resolve()),
        "rows_sampled": int(len(sample)),
        "columns": columns,
        "role_suggestions_require_human_confirmation": suggestions,
        "warning": "Name-based guesses are not a validated mapping. Review sample values and business meaning before using them.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
