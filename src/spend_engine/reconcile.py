from __future__ import annotations

import bisect
from typing import Any

import pandas as pd

from .normalize import (
    normalize_identifier,
    normalize_supplier_name,
    normalize_text,
    parse_number,
    similarity_ratio,
    token_similarity,
)


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


def _supplier_similarity(left: Any, right: Any) -> float:
    """Name similarity after dropping legal suffixes (PTY LTD, GmbH ...)."""
    return similarity_ratio(normalize_supplier_name(left), normalize_supplier_name(right))


def _equivalent_rows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when two ERP rows are interchangeable (same supplier, text and amount).

    Repeated identical receipts (same PO line, different receipt document) are legitimate separate
    rows. A tool invoice that fits all of them can be matched to any one of them with the same
    result, so that is not a real ambiguity.
    """
    a, b = parse_number(left.get("spend_pre_exclusion")), parse_number(right.get("spend_pre_exclusion"))
    if a is None or b is None or abs(a - b) > 0.01:
        return False
    return (
        normalize_supplier_name(left.get("supplier_name", "")) == normalize_supplier_name(right.get("supplier_name", ""))
        and normalize_text(left.get("description", "")) == normalize_text(right.get("description", ""))
    )


def _pick_candidate(
    candidates: list[tuple[float, str]], erp_by_id: dict[str, dict[str, Any]], policy: str
) -> tuple[str | None, bool]:
    """Choose one ERP row. Returns (erp_id or None, was_truly_ambiguous).

    Ties between interchangeable rows pick the lowest record id (deterministic). A tie between
    genuinely different rows still counts the purchase once when policy is "match_first" (the
    total is right either way; only the identity inheritance is skipped); "skip" restores the old
    behaviour of leaving the tool row included, which double counts.
    """
    candidates = sorted(candidates, key=lambda item: (round(item[0], 6), item[1]))
    best_score, best_id = candidates[0]
    tied = [erp_id for score, erp_id in candidates if abs(score - best_score) < 0.01]
    if len(tied) == 1:
        return best_id, False
    if all(_equivalent_rows(erp_by_id[best_id], erp_by_id[other]) for other in tied[1:]):
        return best_id, False
    if policy == "skip":
        return None, True
    return best_id, True


def reconcile_erp_tool(
    records: pd.DataFrame,
    config: dict[str, Any],
    erp_source: str = "ERP",
    tool_source: str = "TOOL",
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    """Match invoice-tool rows to ERP receipts so each purchase is counted once.

    Stage 1  PO (+item) candidates, amount within tolerance, supplier or text evidence.
    Stage 2  one invoice line covering several receipts of one PO item (unique subset sum).
    Stage 3  fallback for tool rows with no PO (or a PO the ERP never saw): same supplier,
             same net amount (tight tolerance), similar text, invoice date near posting date.
    Matched tool rows become DUPLICATE; the ERP row is the one that is counted.
    """
    output = records.copy()
    erp = output.loc[
        (output["source"].astype(str).str.casefold() == erp_source.casefold()) & (output["include"] == 1)
    ].copy()
    tool = output.loc[
        (output["source"].astype(str).str.casefold() == tool_source.casefold()) & (output["include"] == 1)
    ].copy()
    empty_metrics = {
        "matched": 0, "unmatched_tool": len(tool), "ambiguous": 0, "ambiguous_matched": 0,
        "matched_erp_records": 0, "po_matches": 0, "multi_receipt_matches": 0, "fallback_matches": 0,
    }
    if erp.empty or tool.empty:
        return output, {}, empty_metrics

    erp_by_po: dict[str, list[dict[str, Any]]] = {}
    erp_by_id: dict[str, dict[str, Any]] = {}
    for _, row in erp.sort_values("record_id", kind="stable").iterrows():
        row_dict = row.to_dict()
        record_id = str(row_dict["record_id"])
        erp_by_id[record_id] = row_dict
        po = normalize_identifier(row.get("po_number", ""), strip_leading_zeroes=True)
        if po:
            erp_by_po.setdefault(po, []).append(row_dict)

    used_erp: set[str] = set()
    matches: dict[str, str] = {}        # unambiguous: safe to inherit ERP supplier/item keys
    duplicate_tool_ids: set[str] = set()  # every tool row that must be marked DUPLICATE
    ambiguous_unmatched = 0
    ambiguous_matched = 0
    po_matches = 0
    grouped_matches = 0
    fallback_matches = 0
    abs_tol = float(config.get("amount_absolute_tolerance", 2.0))
    rel_tol = float(config.get("amount_relative_tolerance", 0.02))
    supplier_threshold = float(config.get("min_supplier_similarity", 55))
    text_threshold = float(config.get("min_description_similarity", 8))
    policy = str(config.get("ambiguous_policy", "match_first")).lower()
    if config.get("require_unique_best_match") is True and "ambiguous_policy" not in config:
        policy = "match_first"  # legacy flag no longer allowed to cause double counting

    def record_match(tool_id: str, erp_id: str, truly_ambiguous: bool) -> None:
        nonlocal ambiguous_matched
        used_erp.add(erp_id)
        duplicate_tool_ids.add(tool_id)
        if truly_ambiguous:
            ambiguous_matched += 1
        else:
            matches[tool_id] = erp_id

    # ---- Stage 1 and 2: PO based ----
    stage_one_unmatched: list[dict[str, Any]] = []
    for _, tool_series in tool.sort_values("record_id", kind="stable").iterrows():
        tool_row = tool_series.to_dict()
        tool_id = str(tool_row["record_id"])
        po = normalize_identifier(tool_row.get("po_number", ""), strip_leading_zeroes=True)
        if not po or po not in erp_by_po:
            stage_one_unmatched.append(tool_row)  # candidate for the fallback stage
            continue
        tool_item = normalize_identifier(tool_row.get("po_item", ""), strip_leading_zeroes=True)
        eligible_pool: list[dict[str, Any]] = []
        candidates: list[tuple[float, str]] = []
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
            supplier_score = _supplier_similarity(tool_row.get("supplier_name", ""), erp_row.get("supplier_name", ""))
            description_score = 100.0 * token_similarity(
                tool_row.get("description", ""), erp_row.get("description", "")
            )
            supplier_ok = not tool_row.get("supplier_name") or supplier_score >= supplier_threshold
            description_ok = not tool_row.get("description") or description_score >= text_threshold
            if not (supplier_ok or description_ok):
                continue
            candidates.append((amount_delta - (supplier_score * 0.001) - (description_score * 0.001), erp_id))

        if candidates:
            chosen, truly_ambiguous = _pick_candidate(candidates, erp_by_id, policy)
            if chosen is None:
                ambiguous_unmatched += 1
                continue
            record_match(tool_id, chosen, truly_ambiguous)
            po_matches += 1
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
            ambiguous_unmatched += 1
            continue
        if subset:
            selected_rows = [erp_by_id[record_id] for record_id in subset]
            supplier_ok = (
                not tool_row.get("supplier_name")
                or all(
                    _supplier_similarity(tool_row.get("supplier_name", ""), selected.get("supplier_name", ""))
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
            duplicate_tool_ids.add(tool_id)
            used_erp.update(subset)
            grouped_matches += 1

    # ---- Stage 3: fallback without a usable PO ----
    if config.get("fallback_enabled", True) and stage_one_unmatched:
        fb_abs = float(config.get("fallback_amount_absolute_tolerance", 1.0))
        fb_rel = float(config.get("fallback_amount_relative_tolerance", 0.005))
        fb_supplier = float(config.get("fallback_min_supplier_similarity", 85))
        fb_text = float(config.get("fallback_min_description_similarity", 30))
        window_days = float(config.get("fallback_date_window_days", 365))

        amount_index = sorted(
            (parse_number(row.get("spend_pre_exclusion")) or 0.0, erp_id)
            for erp_id, row in erp_by_id.items()
            if (parse_number(row.get("spend_pre_exclusion")) or 0.0) > 0
        )
        amounts_only = [amount for amount, _ in amount_index]

        for tool_row in stage_one_unmatched:
            tool_id = str(tool_row["record_id"])
            target = parse_number(tool_row.get("spend_pre_exclusion"))
            # Without supplier AND text evidence, equal amounts alone are not proof of a duplicate.
            if target is None or target <= 0 or not tool_row.get("supplier_name") or not tool_row.get("description"):
                continue
            tolerance = max(fb_abs, fb_rel * target)
            low = bisect.bisect_left(amounts_only, target - tolerance)
            high = bisect.bisect_right(amounts_only, target + tolerance)
            tool_date = tool_row.get("date_parsed")
            candidates = []
            for amount, erp_id in amount_index[low:high]:
                if erp_id in used_erp:
                    continue
                erp_row = erp_by_id[erp_id]
                supplier_score = _supplier_similarity(tool_row.get("supplier_name", ""), erp_row.get("supplier_name", ""))
                if supplier_score < fb_supplier:
                    continue
                description_score = 100.0 * token_similarity(
                    tool_row.get("description", ""), erp_row.get("description", "")
                )
                if description_score < fb_text:
                    continue
                erp_date = erp_row.get("date_parsed")
                if pd.notna(tool_date) and pd.notna(erp_date) and abs((tool_date - erp_date).days) > window_days:
                    continue
                candidates.append((abs(amount - target) - supplier_score * 0.001 - description_score * 0.001, erp_id))
            if not candidates:
                continue
            chosen, truly_ambiguous = _pick_candidate(candidates, erp_by_id, policy)
            if chosen is None:
                ambiguous_unmatched += 1
                continue
            record_match(tool_id, chosen, truly_ambiguous)
            fallback_matches += 1

    if duplicate_tool_ids:
        mask = output["record_id"].isin(duplicate_tool_ids)
        output.loc[mask, "include"] = 0
        output.loc[mask, "exclusion_reason"] = "DUPLICATE"
    return output, matches, {
        "matched": len(duplicate_tool_ids),
        "unmatched_tool": max(0, len(tool) - len(duplicate_tool_ids)),
        "ambiguous": ambiguous_unmatched,
        "ambiguous_matched": ambiguous_matched,
        "matched_erp_records": len(used_erp),
        "po_matches": po_matches,
        "multi_receipt_matches": grouped_matches,
        "fallback_matches": fallback_matches,
    }
