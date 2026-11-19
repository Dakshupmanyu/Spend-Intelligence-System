"""suggest-exclusions: profile a company file so exclusion rules can be written fast and safely.

It never excludes anything. It ranks document types, account codes and descriptions by how much
they look like non-spend (concept words, mostly negative amounts, blank text) and writes a CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import resolve_path
from .normalize import normalize_text, parse_number
from .pipeline import prepare_records
from .spend import contains_phrase

PROFILE_FIELDS = ("document_type", "account_code", "supplier_id")


def _load_concepts(config: dict[str, Any], config_dir: Path) -> dict[str, list[str]]:
    name = config.get("exclusions", {}).get("concepts_file", "exclusion_concepts.yaml")
    path = resolve_path(config_dir, name)
    if path is None or not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {key: [normalize_text(v) for v in values] for key, values in (loaded.get("concepts") or {}).items()}


def suggest_exclusions(config: dict[str, Any], config_dir: Path, output_dir: Path) -> pd.DataFrame:
    prepared = prepare_records(config, config_dir, disable_label_memory=True)
    records = prepared["records"].copy()
    concepts = _load_concepts(config, config_dir)
    cleaning = config.get("cleaning", {})
    native = records["amount_raw"].map(
        lambda v: parse_number(v, decimal=cleaning.get("numeric_decimal", "."), thousands=cleaning.get("thousands_separator", ","))
    )
    records["_negative"] = native.fillna(0) < 0
    records["_blank_text"] = records["description"].astype(str).str.strip() == ""
    records["_abs_spend"] = records["spend_pre_exclusion"].abs()

    rows: list[dict[str, Any]] = []
    for field in PROFILE_FIELDS:
        if field not in records.columns or (records[field].astype(str).str.strip() == "").all():
            continue
        for value, group in records.groupby(records[field].astype(str).str.strip(), sort=False):
            if not value:
                continue
            samples = group["description"].astype(str).str.strip()
            samples = samples[samples != ""].drop_duplicates().head(3).tolist()
            text = normalize_text(" ".join([value, *group["account_description"].astype(str).head(3), *samples]))
            hits = sorted({name for name, words in concepts.items() if any(contains_phrase(text, w) for w in words)})
            negative_share = float(group["_negative"].mean())
            blank_share = float(group["_blank_text"].mean())
            score = (2 if hits else 0) + (1 if negative_share > 0.5 else 0) + (1 if blank_share > 0.9 and len(group) > 20 else 0)
            rows.append({
                "field": field, "value": value, "rows": len(group), "abs_spend": round(float(group["_abs_spend"].sum()), 2),
                "negative_share": round(negative_share, 3), "blank_text_share": round(blank_share, 3),
                "concept_hits": ",".join(hits), "already_excluded": int((group["include"] == 0).sum()),
                "samples": " | ".join(samples), "suspicion": score,
            })
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["suspicion", "abs_spend"], ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"exclusion_suggestions__{config['company']['id']}.csv"
    report.to_csv(path, index=False)
    report.attrs["path"] = str(path)
    report.attrs["reversal_pairs"] = prepared["reversal_pairs"]
    return report
