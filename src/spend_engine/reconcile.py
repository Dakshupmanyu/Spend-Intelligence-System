from __future__ import annotations

from typing import Any

import pandas as pd

from .normalize import normalize_identifier, normalize_text, parse_number, similarity_ratio, token_similarity


def deduplicate_tool_exports(records: pd.DataFrame, source_name: str = "TOOL") -> tuple[pd.DataFrame, int]:
    output = records.copy()
    mask = (output["source"].astype(str).str.casefold() == source_name.casefold()) & (output["include"] == 1)
    seen: set[tuple[str, str, str, str]] = set()
    duplicate_ids: list[str] = []
    for _, row in output.loc[mask].iterrows():
        invoice = normalize_identifier(row.get("invoice_number", ""))
        supplier = normalize_text(row.get("supplier_name", ""))
        amount = str(parse_number(row.get("amount_raw", "")) or "")
        description = normalize_text(row.get("description", ""))
        quantity = normalize_text(row.get("quantity", ""))
        # Include line text and quantity so different invoice lines are not collapsed.
        key = (invoice, supplier, amount, description + "|" + quantity)
        if invoice and key in seen:
            duplicate_ids.append(str(row["record_id"]))
        elif invoice:
            seen.add(key)
    output.loc[output["record_id"].isin(duplicate_ids), "include"] = 0
    output.loc[output["record_id"].isin(duplicate_ids), "exclusion_reason"] = "DUPLICATE"
    return output, len(duplicate_ids)


def _amount_close(left: Any, right: Any, absolute_tolerance: float, relative_tolerance: float) -> tuple[bool, float]:
    a = parse_number(left)
    b = parse_number(right)
    if a is None or b is None:
        return False, 1e9
    difference = abs(a - b)
    tolerance = max(absolute_tolerance, relative_tolerance * max(abs(a), abs(b), 1.0))
    return difference <= tolerance, difference


def _unique_subset_sum(
    candidates: list[dict[str, Any]], target: float, tolerance: float, max_candidates: int = 24, max_states: int = 30000
) -> tuple[list[str] | None, bool]:
    """Find one unique multi-receipt subset. Returns (ids, ambiguous_or_too_complex)."""
    if len(candidates) < 2 or len(candidates) > max_candidates or target <= 0:
        return None, False
    values: list[tuple[str, int]] = []
    for row in candidates:
        amount = parse_number(row.get("spend_pre_exclusion"))
        if amount is None or amount <= 0:
            continue
        values.append((str(row["record_id"]), int(round(amount * 100))))
    if len(values) < 2:
        return None, False

    target_cents = int(round(target * 100))
    tolerance_cents = max(1, int(round(tolerance * 100)))
    # Sum -> at most two combinations. Two is enough to establish ambiguity.
    states: dict[int, list[tuple[int, ...]]] = {0: [()]}
    for index, (_, value) in enumerate(values):
        additions: dict[int, list[tuple[int, ...]]] = {}
        for subtotal, combinations in states.items():
            new_total = subtotal + value
            if new_total > target_cents + tolerance_cents:
                continue
            for combination in combinations:
                candidate_combination = combination + (index,)
                bucket = additions.setdefault(new_total, [])
                if candidate_combination not in bucket and len(bucket) < 2:
                    bucket.append(candidate_combination)
        for subtotal, combinations in additions.items():
            bucket = states.setdefault(subtotal, [])
            for combination in combinations:
                if combination not in bucket and len(bucket) < 2:
                    bucket.append(combination)
        if len(states) > max_states:
            return None, True

    matches: list[tuple[int, ...]] = []
    for subtotal, combinations in states.items():
        if subtotal == 0 or abs(subtotal - target_cents) > tolerance_cents:
            continue
        for combination in combinations:
            if len(combination) >= 2:
                matches.append(combination)
                if len(matches) > 1:
                    return None, True
    if not matches:
        return None, False
    return [values[index][0] for index in matches[0]], False


def reconcile_erp_tool(
    records: pd.DataFrame,
    config: dict[str, Any],
    erp_source: str = "ERP",
    tool_source: str = "TOOL",
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    """Conservative PO reconciliation with one-to-one and unique multi-receipt matching."""
    output = records.copy()
    erp = output.loc[
        (output["source"].astype(str).str.casefold() == erp_source.casefold()) & (output["include"] == 1)
    ].copy()
    tool = output.loc[
        (output["source"].astype(str).str.casefold() == tool_source.casefold()) & (output["include"] == 1)
    ].copy()
    if erp.empty or tool.empty:
        return output, {}, {"matched": 0, "unmatched_tool": len(tool), "ambiguous": 0, "matched_erp_records": 0}

    erp_by_po: dict[str, list[dict[str, Any]]] = {}
    erp_by_id: dict[str, dict[str, Any]] = {}
    for _, row in erp.iterrows():
        row_dict = row.to_dict()
        record_id = str(row_dict["record_id"])
        erp_by_id[record_id] = row_dict
        po = normalize_identifier(row.get("po_number", ""), strip_leading_zeroes=True)
        if po:
            erp_by_po.setdefault(po, []).append(row_dict)

    used_erp: set[str] = set()
    matches: dict[str, str] = {}
    ambiguous = 0
    grouped_matches = 0
    abs_tol = float(config.get("amount_absolute_tolerance", 2.0))
    rel_tol = float(config.get("amount_relative_tolerance", 0.02))
    supplier_threshold = float(config.get("min_supplier_similarity", 55))
    text_threshold = float(config.get("min_description_similarity", 8))

    for _, tool_row in tool.iterrows():
        po = normalize_identifier(tool_row.get("po_number", ""), strip_leading_zeroes=True)
        if not po or po not in erp_by_po:
            continue
        tool_item = normalize_identifier(tool_row.get("po_item", ""), strip_leading_zeroes=True)
        eligible_pool: list[dict[str, Any]] = []
        candidates = []
        for erp_row in erp_by_po[po]:
            erp_id = str(erp_row["record_id"])
            if erp_id in used_erp:
                continue
            erp_item = normalize_identifier(erp_row.get("po_item", ""), strip_leading_zeroes=True)
            if tool_item and erp_item and tool_item != erp_item:
                continue
            eligible_pool.append(erp_row)

            close, amount_delta = _amount_close(
                tool_row.get("spend_pre_exclusion"), erp_row.get("spend_pre_exclusion"), abs_tol, rel_tol
            )
            if not close:
                continue
            supplier_score = similarity_ratio(tool_row.get("supplier_name", ""), erp_row.get("supplier_name", ""))
            description_score = 100.0 * token_similarity(
                tool_row.get("description", ""), erp_row.get("description", "")
            )
            supplier_ok = not tool_row.get("supplier_name") or supplier_score >= supplier_threshold
            description_ok = not tool_row.get("description") or description_score >= text_threshold
            if not (supplier_ok or description_ok):
                continue
            score = amount_delta - (supplier_score * 0.001) - (description_score * 0.001)
            candidates.append((score, erp_id))

        tool_id = str(tool_row["record_id"])
        if candidates:
            candidates.sort(key=lambda item: item[0])
            if config.get("require_unique_best_match", True) and len(candidates) > 1:
                if abs(candidates[0][0] - candidates[1][0]) < 0.01:
                    ambiguous += 1
                    continue
            erp_id = candidates[0][1]
            used_erp.add(erp_id)
            matches[tool_id] = erp_id
            continue

        # A single invoice line may cover multiple goods receipts for one PO item.
        # Only attempt this when the tool has an explicit item number; PO-wide aggregation is unsafe.
        if not tool_item:
            continue
        aggregate_pool = [
            candidate for candidate in eligible_pool
            if normalize_identifier(candidate.get("po_item", ""), strip_leading_zeroes=True) == tool_item
        ]
        target_amount = parse_number(tool_row.get("spend_pre_exclusion"))
        if target_amount is None or len(aggregate_pool) < 2:
            continue
        tolerance = max(abs_tol, rel_tol * max(abs(target_amount), 1.0))
        subset, subset_ambiguous = _unique_subset_sum(aggregate_pool, target_amount, tolerance)
        if subset_ambiguous:
            ambiguous += 1
            continue
        if subset:
            selected_rows = [erp_by_id[record_id] for record_id in subset]
            supplier_ok = (
                not tool_row.get("supplier_name")
                or all(
                    similarity_ratio(tool_row.get("supplier_name", ""), selected.get("supplier_name", ""))
                    >= supplier_threshold
                    for selected in selected_rows
                )
            )
            description_ok = (
                not tool_row.get("description")
                or all(
                    100.0 * token_similarity(tool_row.get("description", ""), selected.get("description", ""))
                    >= text_threshold
                    for selected in selected_rows
                )
            )
            if not (supplier_ok or description_ok):
                continue
            matches[tool_id] = subset[0]
            used_erp.update(subset)
            grouped_matches += 1

    if matches:
        output.loc[output["record_id"].isin(matches), "include"] = 0
        output.loc[output["record_id"].isin(matches), "exclusion_reason"] = "DUPLICATE"
    return output, matches, {
        "matched": len(matches),
        "unmatched_tool": max(0, len(tool) - len(matches)),
        "ambiguous": ambiguous,
        "matched_erp_records": len(used_erp),
        "multi_receipt_matches": grouped_matches,
    }
