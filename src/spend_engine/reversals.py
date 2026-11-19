"""Detect invoice reversals: an invoice line and its equal-and-opposite credit/reversal.

Example (company B): B000001 +/-13,827.64 on the same invoice, same line, same vendor. Both rows
are real ledger lines but together they spend nothing, so counting either one as a purchase would
overstate spend and the line count. Policy (config `reversals.policy`):

  cancel_both   (default) both rows of each exact pair become CANCELLED with spend 0
  keep          do nothing (net spend is already zero, rows stay included)

Only exact pairs are touched: same supplier, invoice number, invoice line and currency, equal
absolute native amount, opposite sign. A lone negative row (a genuine credit note with no
original in the data) is left included with its negative spend.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .normalize import normalize_text, parse_number


def detect_reversals(
    records: pd.DataFrame, config: dict[str, Any], cleaning: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, int]:
    """Return (records with reversal pairs cancelled, number of pairs found)."""
    policy = str(config.get("policy", "cancel_both")).lower()
    if not config.get("enabled", False) or policy == "keep" or records.empty:
        return records, 0
    cleaning = cleaning or {}
    decimal = cleaning.get("numeric_decimal", ".")
    thousands = cleaning.get("thousands_separator", ",")
    key_columns = list(config.get("key_columns", ["supplier_id", "invoice_number", "invoice_line", "currency"]))
    source_filter = {str(v).casefold() for v in config.get("sources", [])}

    output = records.copy()
    groups: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for _, row in output.loc[output["include"] == 1].sort_values("record_id", kind="stable").iterrows():
        if source_filter and str(row.get("source", "")).casefold() not in source_filter:
            continue
        invoice = normalize_text(row.get("invoice_number", ""))
        if not invoice:  # no invoice number means no evidence that two rows belong together
            continue
        amount = parse_number(row.get("amount_raw", ""), decimal=decimal, thousands=thousands)
        if amount is None or round(abs(amount), 2) == 0:
            continue
        key = (str(row.get("source", "")),) + tuple(normalize_text(row.get(column, "")) for column in key_columns) + (
            round(abs(amount), 2),
        )
        groups[key]["pos" if amount > 0 else "neg"].append(str(row["record_id"]))

    cancelled: list[str] = []
    pairs = 0
    for sides in groups.values():
        count = min(len(sides["pos"]), len(sides["neg"]))
        if count:
            pairs += count
            cancelled.extend(sides["pos"][:count])
            cancelled.extend(sides["neg"][:count])
    if cancelled:
        mask = output["record_id"].astype(str).isin(cancelled)
        output.loc[mask, "include"] = 0
        output.loc[mask, "exclusion_reason"] = "CANCELLED"
        output.loc[mask, "spend_pre_exclusion"] = 0.0
    return output, pairs
