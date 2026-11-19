"""Beam search, look-ahead, ragged trees, blank-text prior, and label memory in the classifier."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from spend_engine.classify import classify_records

COLUMNS = ["l1", "l2", "l3", "l4"]
CFG = {"use_embeddings": False, "use_label_memory": True, "label_memory_weight": 0.35}


def _tax(rows):
    return pd.DataFrame(rows, columns=COLUMNS[: len(rows[0])])


def _records(texts, **extra):
    frame = pd.DataFrame({"record_id": [f"R{i}" for i in range(len(texts))], "description": texts})
    for key, values in extra.items():
        frame[key] = values
    return frame


def _classify(records, taxonomy, columns, examples=None, config=None):
    out, warnings = classify_records(records, taxonomy, columns, examples, {**CFG, **(config or {})}, Path("."))
    return out, warnings


def test_lookahead_recovers_when_upper_level_names_say_nothing():
    # L1/L2 names are meaningless; only the leaf names carry the vocabulary.
    tax = _tax([
        ("Group Alpha", "Class One", "Hydraulic Pumps"),
        ("Group Alpha", "Class One", "Conveyor Belts"),
        ("Group Beta", "Class Two", "Roof Bolts"),
        ("Group Beta", "Class Two", "Safety Boots"),
    ])
    out, _ = _classify(_records(["roof bolts 2m", "conveyor belts steel", "safety boots size 9"]), tax, COLUMNS[:3])
    assert out[["l1", "l3"]].values.tolist() == [
        ["Group Beta", "Roof Bolts"], ["Group Alpha", "Conveyor Belts"], ["Group Beta", "Safety Boots"],
    ]


def test_greedy_would_fail_but_beam_recovers_from_a_misleading_l1_name():
    # The L1 name "Pumps" overlaps the text, but the right leaf sits under L1 "Spares".
    tax = _tax([
        ("Pumps", "Marketing", "Brand Posters"),
        ("Spares", "Mechanical", "Pump Seal Kits"),
    ])
    out, _ = _classify(_records(["pump seal kits for site"]), tax, COLUMNS[:3])
    assert out.loc[0, ["l1", "l3"]].tolist() == ["Spares", "Pump Seal Kits"]


def test_ragged_tree_with_leaf_at_level_two_and_four_levels():
    tax = _tax([
        ("Tools", "Hand Tools", "", ""),
        ("Tools", "Power Tools", "Drills", "Cordless Drills"),
        ("Tools", "Power Tools", "Drills", "Corded Drills"),
        ("Services", "Cleaning", "", ""),
    ])
    tax = tax.replace("", None)
    # taxonomy loader normally stores blanks as ""
    tax = tax.fillna("")
    out, _ = _classify(_records(["cordless drills 18v", "office cleaning", "hand tools set"]), tax, COLUMNS)
    assert out.loc[0, COLUMNS].tolist() == ["Tools", "Power Tools", "Drills", "Cordless Drills"]
    assert out.loc[1, COLUMNS].tolist() == ["Services", "Cleaning", "", ""]
    assert out.loc[2, COLUMNS].tolist() == ["Tools", "Hand Tools", "", ""]


def test_single_level_taxonomy_works():
    tax = pd.DataFrame({"l1": ["Freight", "Software"]})
    out, _ = _classify(_records(["road freight to port", "software licence"]), tax, ["l1"])
    assert out["l1"].tolist() == ["Freight", "Software"]


def test_label_memory_teaches_words_the_taxonomy_cannot():
    tax = _tax([("Equipment", "Electrical", "Fans"), ("Materials", "Roof", "Mesh")])
    examples = pd.DataFrame({
        "record_id": ["E1", "E2"], "label_path": [("Equipment", "Electrical", "Fans"), ("Materials", "Roof", "Mesh")],
        "label_text": ["zorbax glimble 101", "quilbo transmogrifier 55"],
    })
    out, _ = _classify(_records(["zorbax glimble 777", "quilbo transmogrifier 12"]), tax, COLUMNS[:3], examples)
    assert out["l3"].tolist() == ["Fans", "Mesh"]


def test_blank_text_takes_majority_path_of_same_item_then_supplier_with_low_confidence():
    tax = _tax([("Equipment", "Electrical", "Fans"), ("Materials", "Roof", "Mesh")])
    records = _records(
        ["electrical fans", "electrical fans", "", "", ""],
        item_key=["I1", "I1", "I1", "I9", "I9"], supplier_key=["S1", "S1", "S1", "S1", "S7"],
    )
    out, warnings = _classify(records, tax, COLUMNS[:3])
    assert out.loc[2, "l3"] == "Fans" and 0 < out.loc[2, "confidence"] <= 0.2      # via item I1
    assert out.loc[3, "l3"] == "Fans" and 0 < out.loc[3, "confidence"] <= 0.2      # item I9 unknown -> supplier S1
    assert out.loc[4, "confidence"] == 0.0                                           # nothing to borrow
    assert out.loc[4, ["l1", "l2", "l3"]].ne("").all()                              # levels are still filled
    assert any("no description" in w for w in warnings)


def test_every_included_row_gets_every_level_and_identical_texts_agree():
    tax = _tax([("A1", "B1", "C1"), ("A2", "B2", "C2")])
    out, _ = _classify(_records(["c1 thing", "c1 thing", "nonsense words"]), tax, COLUMNS[:3])
    assert out.loc[0, COLUMNS[:3]].tolist() == out.loc[1, COLUMNS[:3]].tolist()
    assert out[["l1", "l2", "l3"]].ne("").all().all()
    assert out["confidence"].between(0, 1).all()


def test_beam_width_one_equals_greedy_on_an_easy_case():
    tax = _tax([("Equipment", "Electrical", "Fans"), ("Materials", "Roof", "Mesh")])
    out, _ = _classify(_records(["roof mesh sheet"]), tax, COLUMNS[:3], config={"beam_width": 1})
    assert out.loc[0, "l3"] == "Mesh"
