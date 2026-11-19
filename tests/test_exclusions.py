"""#13: reversal pairs, whole-word keywords, status phrases, suggest-exclusions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from spend_engine.config import load_config
from spend_engine.reversals import detect_reversals
from spend_engine.spend import apply_exclusions, contains_phrase, status_is_approved
from spend_engine.suggest import suggest_exclusions

ROOT = Path(__file__).resolve().parents[1]


def _rec(rid, amount, invoice="INV1", line="1", vendor="V1", currency="USD", **extra):
    return {
        "record_id": rid, "source": "B", "include": 1, "exclusion_reason": "", "amount_raw": str(amount),
        "invoice_number": invoice, "invoice_line": line, "supplier_id": vendor, "currency": currency,
        "spend_pre_exclusion": float(amount), **extra,
    }


def _frame(rows):
    return pd.DataFrame(rows)


CFG = {"enabled": True, "policy": "cancel_both"}


def test_exact_offset_pair_is_cancelled_with_zero_spend():
    out, pairs = detect_reversals(_frame([_rec("B1", -13827.64), _rec("B2", 13827.64), _rec("B3", 50)]), CFG)
    out = out.set_index("record_id")
    assert pairs == 1
    assert out.loc[["B1", "B2"], "exclusion_reason"].tolist() == ["CANCELLED", "CANCELLED"]
    assert out.loc[["B1", "B2"], "spend_pre_exclusion"].tolist() == [0.0, 0.0]
    assert out.loc["B3", "include"] == 1


def test_lone_credit_note_stays_included_and_negative():
    out, pairs = detect_reversals(_frame([_rec("B3", -2611.52)]), CFG)
    assert pairs == 0 and out.loc[0, "include"] == 1 and out.loc[0, "spend_pre_exclusion"] == -2611.52


def test_pairing_needs_same_vendor_invoice_line_currency_and_amount():
    rows = [
        _rec("A", 100, vendor="V1"), _rec("B", -100, vendor="V2"),            # other vendor
        _rec("C", 100, invoice="I1"), _rec("D", -100, invoice="I2"),          # other invoice
        _rec("E", 100, line="1"), _rec("F", -100, line="2"),                  # other line
        _rec("G", 100, currency="USD"), _rec("H", -100, currency="EUR"),      # other currency
        _rec("I", 100), _rec("J", -100.5),                                    # other amount
    ]
    out, pairs = detect_reversals(_frame(rows), CFG)
    assert pairs == 0 and (out["include"] == 1).all()


def test_only_matched_count_is_cancelled_when_unbalanced():
    out, pairs = detect_reversals(_frame([_rec("P1", 10), _rec("P2", 10), _rec("N1", -10)]), CFG)
    assert pairs == 1
    assert sorted(out.loc[out["include"] == 0, "record_id"]) == ["N1", "P1"]  # P2 has no partner


def test_no_invoice_number_means_no_pairing_and_keep_policy_disables_it():
    out, pairs = detect_reversals(_frame([_rec("X", 5, invoice=""), _rec("Y", -5, invoice="")]), CFG)
    assert pairs == 0
    out, pairs = detect_reversals(_frame([_rec("X", 5), _rec("Y", -5)]), {"enabled": True, "policy": "keep"})
    assert pairs == 0 and (out["include"] == 1).all()


def test_keywords_match_whole_words_only():
    assert contains_phrase("input tax payable", "tax")
    assert not contains_phrase("syntax converter", "tax")
    assert contains_phrase("royalty payment q1", "royalty")
    assert contains_phrase("partner billing for site", "partner billing")


def test_status_phrases_and_negatives():
    allowed = {"approved", "paid"}
    assert status_is_approved("approved for payment", allowed)
    assert status_is_approved("paid", allowed)
    assert not status_is_approved("not approved", allowed)
    assert not status_is_approved("unapproved", allowed)
    assert not status_is_approved("pending approval", allowed)
    assert not status_is_approved("approval rejected", allowed)
    assert not status_is_approved("", allowed)


def test_status_cancelled_phrase_and_keyword_exclusion_in_apply_exclusions():
    records = pd.DataFrame([
        {"record_id": "1", "source_config_key": "t", "status": "Cancelled by user", "description": "valve", "account_description": "", "document_type": "", "account_code": ""},
        {"record_id": "2", "source_config_key": "t", "status": "Approved for payment", "description": "syntax manual", "account_description": "", "document_type": "", "account_code": ""},
        {"record_id": "3", "source_config_key": "t", "status": "Paid", "description": "VAT payable", "account_description": "", "document_type": "", "account_code": ""},
        {"record_id": "4", "source_config_key": "t", "status": "Not approved", "description": "valve", "account_description": "", "document_type": "", "account_code": ""},
    ])
    sources = {"t": {"status_required": True, "allowed_status_values": ["approved", "paid"]}}
    out = apply_exclusions(records, sources, {"cancelled_status_values": ["cancelled"], "non_procurement_keywords": ["tax", "vat payable"]})
    assert out["exclusion_reason"].tolist() == ["CANCELLED", "", "NON_PROCUREMENT", "NOT_APPROVED"]


def test_suggest_exclusions_ranks_without_excluding(tmp_path):
    config, config_dir = load_config(ROOT / "configs" / "synthetic_b.yaml")
    report = suggest_exclusions(config, config_dir, tmp_path)
    assert (tmp_path / "exclusion_suggestions__B.csv").is_file()
    assert {"field", "value", "rows", "abs_spend", "negative_share", "concept_hits", "suspicion"} <= set(report.columns)
