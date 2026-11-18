from __future__ import annotations

import pandas as pd
import pytest

from spend_engine.canonical import canonicalize_source
from spend_engine.config import ConfigurationError
from spend_engine.spend import build_fx_lookup
from spend_engine.taxonomy import load_taxonomy


def test_blank_record_ids_and_repeated_headers_are_dropped_not_invented():
    raw = pd.DataFrame({"line_id": ["A001", "  ", "line_id"], "amount": ["10", "20", "amount"]})
    source_config = {
        "record_id_column": "line_id",
        "amount_column": "amount",
        "amount_type": "net",
        "source_name": "TEST",
    }
    canonical, warnings = canonicalize_source(raw, "test", source_config, "X")
    assert list(canonical["record_id"]) == ["A001"]  # no ID is ever invented
    assert any("dropped 2 junk row" in warning for warning in warnings)


def test_taxonomy_with_missing_parent_is_rejected(tmp_path):
    path = tmp_path / "taxonomy.csv"
    pd.DataFrame(
        [{"l1": "Operations", "l2": "", "l3": "Spare Parts"}]
    ).to_csv(path, index=False)
    with pytest.raises(ConfigurationError, match="missing parent levels"):
        load_taxonomy(path, ["l1", "l2", "l3"])


def test_conflicting_monthly_fx_rates_are_rejected():
    fx = pd.DataFrame(
        [
            {"currency": "EUR", "month": "2024-01", "units_per_usd": "1.08"},
            {"currency": "EUR", "month": "2024-01", "units_per_usd": "1.10"},
        ]
    )
    config = {
        "currency_column": "currency",
        "month_column": "month",
        "units_per_usd_column": "units_per_usd",
    }
    with pytest.raises(ValueError, match="conflicting"):
        build_fx_lookup(fx, config)
