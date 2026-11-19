from __future__ import annotations

from typing import Any

import pandas as pd

from .normalize import normalize_text, parse_date, parse_number, parse_vat_rate


def build_fx_lookup(fx_df: pd.DataFrame | None, fx_config: dict[str, Any] | None) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    if fx_df is None or not fx_config:
        return lookup
    required = [fx_config.get("currency_column"), fx_config.get("month_column"), fx_config.get("units_per_usd_column")]
    if any(column not in fx_df.columns for column in required):
        raise ValueError(f"FX file must contain configured columns: {required}")
    for _, row in fx_df.iterrows():
        currency = str(row[required[0]]).strip().upper()
        month = str(row[required[1]]).strip()[:7]
        rate = parse_number(row[required[2]])
        if currency and month and rate and rate > 0:
            key = (currency, month)
            if key in lookup and abs(lookup[key] - rate) > 1e-12:
                raise ValueError(
                    f"FX file has conflicting units_per_usd values for {currency} {month}: "
                    f"{lookup[key]} vs {rate}"
                )
            lookup[key] = rate
    return lookup


def _month_index(month: str) -> int | None:
    """Convert 'YYYY-MM' to a month counter so distances between months are simple integers."""
    try:
        year, mon = month[:7].split("-")
        return int(year) * 12 + int(mon) - 1
    except (ValueError, AttributeError):
        return None


def _build_fx_index(lookup: dict[tuple[str, str], float]) -> dict[str, list[tuple[int, float]]]:
    """Group the (currency, month) lookup per currency as a sorted list of (month_index, rate)."""
    index: dict[str, list[tuple[int, float]]] = {}
    for (currency, month), rate in lookup.items():
        position = _month_index(month)
        if position is not None:
            index.setdefault(currency, []).append((position, rate))
    for entries in index.values():
        entries.sort()
    return index


def _lookup_rate(
    index: dict[str, list[tuple[int, float]]], currency: str, date: pd.Timestamp | pd.NaT
) -> tuple[float | None, str]:
    """Return (rate, how). how is 'exact', 'nearest', 'latest' (no usable date) or 'none'.

    A missing month must never abort a run: use the closest month available for that currency
    (earlier month wins a tie). With no usable date, use the currency's latest month.
    Only a currency absent from the FX table entirely returns no rate.
    """
    entries = index.get(currency)
    if not entries:
        return None, "none"
    if pd.isna(date):
        return entries[-1][1], "latest"
    target = int(date.year) * 12 + int(date.month) - 1
    best = min(entries, key=lambda item: (abs(item[0] - target), item[0]))
    return best[1], ("exact" if best[0] == target else "nearest")


def _normalised_date(row: pd.Series, source_config: dict[str, Any]) -> pd.Timestamp | pd.NaT:
    formats = source_config.get("date_formats", [])
    date = parse_date(row.get("date_raw", ""), formats)
    if pd.isna(date):
        fallback = row.get("fallback_date_raw", "") or row.get("fx_date_raw", "")
        date = parse_date(fallback, formats)
    return date


def normalize_amounts(
    records: pd.DataFrame,
    source_configs: dict[str, Any],
    reporting_currency: str,
    fx_lookup: dict[tuple[str, str], float],
    cleaning_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    output = records.copy()
    warnings: list[str] = []
    cleaning = cleaning_config or {}
    decimal = cleaning.get("numeric_decimal", ".")
    thousands = cleaning.get("thousands_separator", ",")
    spend_values: list[float] = []
    normalized_dates: list[pd.Timestamp | pd.NaT] = []
    failed_dates: list[bool] = []
    conversion_errors: list[str] = []
    fx_index = _build_fx_index(fx_lookup)
    fx_approximated = 0  # rows converted with a nearby month / latest month instead of the exact one

    for _, row in output.iterrows():
        source_cfg = source_configs[row["source_config_key"]]
        date = _normalised_date(row, source_cfg)
        normalized_dates.append(date)
        date_is_configured = bool(
            source_cfg.get("date_column") or source_cfg.get("fallback_fx_date_column")
        )
        failed_dates.append(bool(pd.isna(date) and date_is_configured))
        if int(row.get("include", 1)) == 0:
            spend_values.append(0.0)
            continue
        amount = parse_number(row["amount_raw"], decimal=decimal, thousands=thousands)
        if amount is None:
            conversion_errors.append(f"{row['record_id']}: amount is blank or invalid")
            spend_values.append(float("nan"))
            continue

        if row["amount_type"] == "gross":
            vat = parse_vat_rate(row.get("vat_raw", ""), float(source_cfg.get("default_vat_rate", 0.15)))
            amount = amount / (1.0 + vat)

        if source_cfg.get("amount_in_reporting_currency"):
            # e.g. a value_zar column: already in the reporting currency, so no FX and no currency needed.
            spend_values.append(amount)
            continue

        currency_value = str(row.get("currency", "")).strip().upper()
        currency = currency_value or str(source_cfg.get("default_currency", "")).strip().upper()
        if not currency:
            conversion_errors.append(f"{row['record_id']}: currency is blank and no default_currency is configured")
            spend_values.append(float("nan"))
            continue
        if currency != reporting_currency.upper():
            row_rate = parse_number(row.get("row_fx_rate_raw", ""), decimal=decimal, thousands=thousands)
            if currency == "USD" and reporting_currency.upper() == "USD":
                rate = 1.0
            elif row_rate and row_rate > 0:
                rate = row_rate
            else:
                rate, how = _lookup_rate(fx_index, currency, date)
                if rate and how in {"nearest", "latest"}:
                    fx_approximated += 1
            if not rate or rate <= 0:
                conversion_errors.append(
                    f"{row['record_id']}: no valid {currency}/USD rate for {row.get('fx_date_raw', '')}"
                )
                spend_values.append(float("nan"))
                continue
            # FX rates in the task are units of local currency per 1 USD.
            if reporting_currency.upper() == "USD":
                amount = amount / rate
            else:
                conversion_errors.append(
                    f"{row['record_id']}: conversion to {reporting_currency} is not configured"
                )
                spend_values.append(float("nan"))
                continue
        spend_values.append(float(amount))

    # A bad amount or an unconvertible currency never aborts the run: the row stays in the
    # output with spend 0, and the problem is counted and listed in the run report.
    spend_series = pd.Series(spend_values, index=output.index, dtype=float)
    unconverted = int(spend_series.isna().sum())
    output["spend_pre_exclusion"] = spend_series.fillna(0.0)
    output["spend_unconverted"] = spend_series.isna()
    output["date_parsed"] = normalized_dates
    output["date_parse_failed"] = failed_dates
    output.attrs["fx_approximated_rows"] = fx_approximated
    output.attrs["unconverted_rows"] = unconverted
    if fx_approximated:
        warnings.append(
            f"{fx_approximated} row(s) converted with the nearest available FX month "
            "(exact month missing or no usable date)"
        )
    if unconverted:
        warnings.append(
            f"{unconverted} included row(s) could not be converted (blank/invalid amount or currency "
            "missing from FX table); kept with spend=0, see messages below"
        )
    if conversion_errors:
        warnings.extend(conversion_errors[:50])
        if len(conversion_errors) > 50:
            warnings.append(f"{len(conversion_errors) - 50} additional amount/FX errors omitted from summary")
    return output, warnings


def apply_exclusions(records: pd.DataFrame, source_configs: dict[str, Any], exclusion_config: dict[str, Any]) -> pd.DataFrame:
    output = records.copy()
    include_values: list[int] = []
    reasons: list[str] = []
    for _, row in output.iterrows():
        source_cfg = source_configs[row["source_config_key"]]
        status = normalize_text(row.get("status", ""))
        allowed = {normalize_text(v) for v in source_cfg.get("allowed_status_values", ["approved", "paid"])}
        cancelled = {normalize_text(v) for v in exclusion_config.get("cancelled_status_values", [])}
        reason = ""
        cancelled_doc_types = {str(v).strip().casefold() for v in exclusion_config.get("cancelled_document_types", [])}
        document_type = str(row.get("document_type", "")).strip().casefold()
        absent_policy = source_cfg.get("status_absent_policy", "include_with_warning")
        if status and status in cancelled:
            reason = "CANCELLED"
        elif document_type and document_type in cancelled_doc_types:
            reason = "CANCELLED"
        elif source_cfg.get("status_required") and status not in allowed:
            reason = "NOT_APPROVED"
        elif not status and source_cfg.get("status_required"):
            reason = "NOT_APPROVED"
        elif not status and absent_policy == "exclude_unknown":
            reason = "NOT_APPROVED"
        elif not status and absent_policy == "fail":
            raise ValueError(f"{row['record_id']}: status missing and status_absent_policy=fail")
        else:
            haystack = " ".join([
                str(row.get("description", "")),
                str(row.get("account_description", "")),
                str(row.get("document_type", "")),
            ])
            normalized = normalize_text(haystack)
            keywords = [normalize_text(k) for k in exclusion_config.get("non_procurement_keywords", [])]
            if any(keyword and keyword in normalized for keyword in keywords):
                reason = "NON_PROCUREMENT"
            codes = {str(code).strip().casefold() for code in exclusion_config.get("account_code_exclusions", [])}
            if not reason and str(row.get("account_code", "")).strip().casefold() in codes and codes:
                reason = "NON_PROCUREMENT"
            doc_types = {str(value).strip().casefold() for value in exclusion_config.get("document_type_exclusions", [])}
            if not reason and str(row.get("document_type", "")).strip().casefold() in doc_types and doc_types:
                reason = "NON_PROCUREMENT"
        include_values.append(0 if reason else 1)
        reasons.append(reason)
    output["include"] = include_values
    output["exclusion_reason"] = reasons
    return output
