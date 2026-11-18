"""Regression tests for the four prod crashers: FX gaps, FX date column, embedding model, bad CSV rows."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from spend_engine.classify import Embedder
from spend_engine.cli import main
from spend_engine.io import read_table
from spend_engine.spend import _build_fx_index, _lookup_rate

REPO = Path(__file__).resolve().parents[1]

B_HEADER = (
    "line_id,source_system,region,invoice_number,invoice_line,posting_date,invoice_date,due_date,paid_date,"
    "line_description,material_number,quantity,currency,currency_rate,line_amount,cost_centre,gl_account,"
    "gl_account_description,vendor_id,supplier_type,document_type,po_number,po_line,po_type,plant,"
    "company_code,payment_terms,transport_flag"
)
B_ROWS = [
    # USD, invoice_date years before the FX table: must not matter
    "B000001,ERP-B1 (JDE),R1,B-INV-1,1,3/29/2024,7/23/2018,3/29/2024,,,MAT-1,0,USD,,-13827.64,CC1,GL40007,BLCN,V500009,,PD,,0,,PL1,CO102,,",
    # HKD with no row rate and FX table has no 2024-01 / 2023-08 for it nearby exactly
    "B000004,ERP-B1 (JDE),R1,B-INV-2,1,1/15/2024,8/1/2023,8/1/2023,1/17/2024,,MAT-1,0,HKD,,1507.39,CC1,GL40012,VAPN,V500014,,PV,,0,,PL1,CO106,PP,",
    # Currency that is not in the FX table at all
    "B000005,ERP-B1 (JDE),R1,B-INV-3,1,3/29/2024,9/26/2023,10/26/2023,3/29/2024,,MAT-2,240,XYZ,,5529.29,CC2,GL40007,VAPN,V500019,,PV,PO-1,1,OP,PL2,CO102,AA,",
    "B000007,ERP-B1 (JDE),R1,B-INV-4,1,1/30/2024,11/24/2023,12/24/2023,2/27/2024,INK CODER VISUAL YEL 1 QT,MAT-3,2,MMK,,656.02,CC2,GL40007,VAPI,V500026,,PV,PO-2,1,OP,PL2,CO102,N30,",
]


def _write_b_project(root: Path, extra_lines: list[str] | None = None) -> Path:
    (root / "configs").mkdir()
    data = root / "data" / "company_b"
    data.mkdir(parents=True)
    lines = [B_HEADER, *B_ROWS, *(extra_lines or [])]
    (data / "invoice_lines.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data / "fx_rates_usd.csv").write_text(
        "currency,month,units_per_usd\nHKD,2024-03,7.8\nMMK,2023-11,2100\nMMK,2024-04,2100\n", encoding="utf-8"
    )
    (data / "taxonomy_B.csv").write_text(
        "category_l1,category_l2,category_l3\nDirect Materials,Coatings & Inks,Inks & Varnishes\n"
        "Indirect Materials,MRO,Filters\n",
        encoding="utf-8",
    )
    (data / "seed_labels.csv").write_text(
        "line_id,include,exclusion_reason,category_l1,category_l2,category_l3\n", encoding="utf-8"
    )
    config_text = (REPO / "configs" / "company_b.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace("use_embeddings: true", "use_embeddings: false")
    config_path = root / "configs" / "company_b.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def test_fx_lookup_exact_nearest_latest_none():
    index = _build_fx_index({("HKD", "2024-03"): 7.8, ("HKD", "2024-06"): 7.7})
    assert _lookup_rate(index, "HKD", pd.Timestamp("2024-03-10")) == (7.8, "exact")
    assert _lookup_rate(index, "HKD", pd.Timestamp("2024-01-15")) == (7.8, "nearest")
    assert _lookup_rate(index, "HKD", pd.Timestamp("2024-05-02")) == (7.7, "nearest")
    # tie (April is one month from March and two from June -> March; use exact tie: 2024-04.5 n/a) -> earlier wins
    tie_index = _build_fx_index({("EUR", "2024-01"): 0.9, ("EUR", "2024-03"): 0.8})
    assert _lookup_rate(tie_index, "EUR", pd.Timestamp("2024-02-01")) == (0.9, "nearest")
    assert _lookup_rate(index, "HKD", pd.NaT) == (7.7, "latest")
    assert _lookup_rate(index, "XYZ", pd.Timestamp("2024-03-10")) == (None, "none")


def test_b_run_survives_missing_fx_months_and_unknown_currency(tmp_path):
    config_path = _write_b_project(tmp_path)
    code = main(["run", "--config", str(config_path), "--output-dir", str(tmp_path / "out")])
    assert code == 0
    out = pd.read_csv(tmp_path / "out" / "candidate__B.csv", dtype={"record_id": str})
    assert len(out) == len(B_ROWS)
    by_id = out.set_index("record_id")
    # HKD 1507.39 / 7.8 via the nearest month (FX table only has 2024-03)
    assert by_id.loc["B000004", "spend"] == pytest.approx(1507.39 / 7.8, abs=0.01)
    # unknown currency: row kept, spend 0, flagged in the report, no abort
    assert by_id.loc["B000005", "spend"] == 0
    report = (tmp_path / "out" / "candidate__B__run_report.json").read_text(encoding="utf-8")
    assert '"included_rows_unconverted_spend_zero": 1' in report


def test_fx_month_comes_from_posting_date_not_ancient_invoice_date(tmp_path):
    config_path = _write_b_project(tmp_path)
    assert main(["run", "--config", str(config_path), "--output-dir", str(tmp_path / "out")]) == 0
    out = pd.read_csv(tmp_path / "out" / "candidate__B.csv", dtype={"record_id": str}).set_index("record_id")
    # B000007: MMK, posting 2024-01 (invoice 2023-11 also exists). Posting-date month 2024-01 is missing, so
    # nearest is 2023-11 (2 months) vs 2024-04 (3 months) -> 2100
    assert out.loc["B000007", "spend"] == pytest.approx(656.02 / 2100, abs=0.01)  # output rounds to 2 dp


def test_bad_lines_header_repeat_and_blank_ids_do_not_abort(tmp_path):
    junk = [
        "B000099,ERP-B1 (JDE),R1,too,many,columns," + ",".join(["x"] * 40),  # malformed width
        B_HEADER,  # repeated header row mid-file
        ",ERP-B1 (JDE),R1,B-INV-9,1,1/1/2024,1/1/2024,,,,MAT-9,1,USD,,10,CC1,GL1,X,V1,,PV,,0,,PL1,CO1,,",  # blank id
    ]
    config_path = _write_b_project(tmp_path, junk)
    code = main(["run", "--config", str(config_path), "--output-dir", str(tmp_path / "out")])
    assert code == 0
    out = pd.read_csv(tmp_path / "out" / "candidate__B.csv", dtype={"record_id": str})
    assert sorted(out["record_id"]) == sorted(["B000001", "B000004", "B000005", "B000007"])


def test_read_table_counts_skipped_lines_and_falls_back_on_encoding(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes("a,b\n1,2\n3,4,5\n6,caf\xe9\n".encode("latin-1"))
    frame = read_table(path, {"delimiter": ","})
    assert frame.attrs["bad_lines_skipped"] == 1
    assert list(frame["a"]) == ["1", "6"]
    assert frame.loc[1, "b"] == "café"


def test_embedder_falls_back_when_model_cannot_load(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` raise ImportError
    embedder = Embedder({"use_embeddings": True}, tmp_path)
    assert not embedder.available
    assert embedder.error
    assert embedder.encode(["anything"]) is None


def test_embedder_strict_mode_still_raises_when_requested(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(RuntimeError):
        Embedder({"use_embeddings": True, "fallback_to_lexical": False}, tmp_path)
