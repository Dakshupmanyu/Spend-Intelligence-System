"""Regression tests: a purchase present in both ERP and invoice tool must be counted exactly once."""
from __future__ import annotations

import pandas as pd

from spend_engine.reconcile import reconcile_erp_tool

CFG = {
    "amount_absolute_tolerance": 2.0,
    "amount_relative_tolerance": 0.02,
    "min_supplier_similarity": 55,
    "min_description_similarity": 8,
}
FAN = "FAN ELEC;22KW,YARVEX FLW,DIA 760MM,FLNG"
MESH = "MESH;WLD,3M,WD:2,2M,100 X 75MM,4MM"


def _row(rid, source, po, amount, supplier, text, date="2021-06-28", item=""):
    return {
        "record_id": rid, "source": source, "include": 1, "exclusion_reason": "",
        "po_number": po, "po_item": item, "spend_pre_exclusion": amount,
        "supplier_name": supplier, "description": text, "date_parsed": pd.Timestamp(date),
    }


def _run(rows, **overrides):
    out, matches, metrics = reconcile_erp_tool(pd.DataFrame(rows), {**CFG, **overrides}, "ERP", "TOOL")
    return out.set_index("record_id"), matches, metrics


def test_identical_erp_receipts_do_not_block_the_match():
    rows = [
        _row("ERP:A2", "ERP", "4510000018", 104243.4, "YARVEX", FAN, item="10"),
        _row("ERP:A3", "ERP", "4510000018", 104243.4, "YARVEX", FAN, item="10"),
        _row("TOOL:T1", "TOOL", "4510000018", 104243.4, "Yarvex", FAN, item="10"),
    ]
    out, matches, metrics = _run(rows)
    assert out.loc["TOOL:T1", "exclusion_reason"] == "DUPLICATE" and out.loc["TOOL:T1", "include"] == 0
    assert out.loc[["ERP:A2", "ERP:A3"], "include"].tolist() == [1, 1]  # both receipts still counted
    assert matches == {"TOOL:T1": "ERP:A2"}  # deterministic: lowest record id
    assert metrics["ambiguous"] == 0


def test_two_tool_invoices_for_two_identical_receipts_consume_one_each():
    rows = [
        _row("ERP:A2", "ERP", "45", 500.0, "YARVEX", FAN),
        _row("ERP:A3", "ERP", "45", 500.0, "YARVEX", FAN),
        _row("TOOL:T1", "TOOL", "45", 500.0, "Yarvex", FAN),
        _row("TOOL:T2", "TOOL", "45", 500.0, "Yarvex", FAN),
        _row("TOOL:T3", "TOOL", "45", 500.0, "Yarvex", FAN),  # third invoice has no receipt left
    ]
    out, matches, _ = _run(rows)
    assert out.loc[["TOOL:T1", "TOOL:T2"], "include"].tolist() == [0, 0]
    assert out.loc["TOOL:T3", "include"] == 1  # genuinely tool-only spend
    assert sorted(matches.values()) == ["ERP:A2", "ERP:A3"]


def test_tool_row_without_po_matches_on_supplier_amount_and_text():
    rows = [
        _row("ERP:A6", "ERP", "4510000042", 99180.0, "PRATON (PTY) LTD", MESH),
        _row("TOOL:T2", "TOOL", "", 99180.0, "PRATON PTY LTD", MESH, date="2021-07-15"),
    ]
    out, matches, metrics = _run(rows)
    assert out.loc["TOOL:T2", "exclusion_reason"] == "DUPLICATE"
    assert metrics["fallback_matches"] == 1 and matches["TOOL:T2"] == "ERP:A6"


def test_fallback_does_not_merge_different_supplier_text_or_far_dates():
    erp = _row("ERP:A6", "ERP", "4510000042", 99180.0, "PRATON (PTY) LTD", MESH)
    cases = {
        "other supplier": _row("TOOL:T", "TOOL", "", 99180.0, "ZANMORE HYDRAULIC", MESH),
        "other text": _row("TOOL:T", "TOOL", "", 99180.0, "PRATON PTY LTD", "SERVICE;ROOF BOLTER OVERHAUL"),
        "other amount": _row("TOOL:T", "TOOL", "", 90000.0, "PRATON PTY LTD", MESH),
        "years apart": _row("TOOL:T", "TOOL", "", 99180.0, "PRATON PTY LTD", MESH, date="2024-09-01"),
        "no text": _row("TOOL:T", "TOOL", "", 99180.0, "PRATON PTY LTD", ""),
    }
    for name, tool in cases.items():
        out, matches, _ = _run([erp, tool])
        assert out.loc["TOOL:T", "include"] == 1, name
        assert matches == {}, name


def test_fallback_skipped_when_tool_po_exists_in_erp_but_nothing_fits():
    rows = [
        _row("ERP:A1", "ERP", "4510000018", 1000.0, "YARVEX", FAN),
        _row("ERP:A9", "ERP", "4510000099", 500.0, "YARVEX", FAN),
        _row("TOOL:T1", "TOOL", "4510000018", 500.0, "Yarvex", FAN),  # PO known, amount differs
    ]
    out, _, _ = _run(rows)
    assert out.loc["TOOL:T1", "include"] == 1  # must not jump to ERP:A9 through the fallback


def test_genuinely_ambiguous_still_counts_once_but_skips_identity_inheritance():
    rows = [
        _row("ERP:A1", "ERP", "45", 499.5, "YARVEX", FAN),  # equally close to the invoice ...
        _row("ERP:A2", "ERP", "45", 500.5, "YARVEX", FAN),  # ... but a different amount: not interchangeable
        _row("TOOL:T1", "TOOL", "45", 500.0, "Yarvex", FAN),
    ]
    out, matches, metrics = _run(rows)
    assert out.loc["TOOL:T1", "exclusion_reason"] == "DUPLICATE"
    assert matches == {}  # ambiguous: no key inheritance
    assert metrics["ambiguous_matched"] == 1
    skipped, _, skipped_metrics = _run(rows, ambiguous_policy="skip")
    assert skipped.loc["TOOL:T1", "include"] == 1 and skipped_metrics["ambiguous"] == 1
