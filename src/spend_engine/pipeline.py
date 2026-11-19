from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .canonical import canonicalize_source
from .classify import classify_records
from .config import resolve_path
from .identity import assign_entity_keys, inherit_matched_identity
from .io import read_table
from .labels import attach_labels, label_identity_overrides
from .memory import build_memory_examples
from .normalize import normalize_identifier
from .reconcile import deduplicate_tool_exports, reconcile_erp_tool
from .spend import apply_exclusions, build_fx_lookup, normalize_amounts
from .taxonomy import load_taxonomy
from .validate import OUTPUT_COLUMNS, validate_output

LOGGER = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


def _apply_identity_overrides(
    records: pd.DataFrame,
    supplier_overrides: dict[tuple[str, str], str],
    item_overrides: dict[tuple[str, str], str],
) -> pd.DataFrame:
    output = records.copy()
    for index, row in output.iterrows():
        supplier_id = normalize_identifier(row.get("supplier_id", ""))
        item_id = normalize_identifier(row.get("item_id", ""))
        supplier_key = supplier_overrides.get((str(row.get("source_namespace", "")), supplier_id)) if supplier_id else None
        item_key = item_overrides.get((str(row.get("item_namespace", row.get("source", ""))), item_id)) if item_id else None
        if supplier_key:
            output.at[index, "supplier_key"] = supplier_key
        if item_key:
            output.at[index, "item_key"] = item_key
    return output


def _make_output(records: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = pd.DataFrame(index=records.index)
    output["record_id"] = records["record_id"].astype(str)
    output["include"] = records["include"].astype(int)
    output["exclusion_reason"] = records["exclusion_reason"].fillna("").astype(str)
    output["supplier_key"] = records["supplier_key"].fillna("").astype(str)
    output["item_key"] = records["item_key"].fillna("").astype(str)
    output["spend"] = pd.to_numeric(records["spend_pre_exclusion"], errors="coerce").fillna(0.0)
    output.loc[output["include"] == 0, "spend"] = 0.0
    for depth in range(1, 5):
        column = f"l{depth}"
        output[column] = records[column].fillna("").astype(str) if column in records else ""
        output.loc[output["include"] == 0, column] = ""
    output["confidence"] = pd.to_numeric(records.get("confidence", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    spend_decimals = int(config.get("output", {}).get("spend_decimals", 2))
    confidence_decimals = int(config.get("output", {}).get("confidence_decimals", 4))
    output["spend"] = output["spend"].round(spend_decimals)
    output["confidence"] = output["confidence"].round(confidence_decimals)
    return output[OUTPUT_COLUMNS]


def prepare_records(
    config: dict[str, Any],
    config_dir: Path,
    disable_label_memory: bool = False,
    attach_run_labels: bool = False,
) -> dict[str, Any]:
    """Everything up to (not including) classification.

    attach_run_labels=True is for the evaluate command only: it attaches the golden labels to the
    run rows as an answer key. A normal run never does that.
    """
    warnings: list[str] = []
    files_cfg = config["files"]
    csv_cfg = config.get("csv", {})
    source_frames: list[pd.DataFrame] = []
    source_configs = config["sources"]

    for source_key, source_cfg in source_configs.items():
        file_path = resolve_path(config_dir, files_cfg.get(source_key))
        if file_path is None:
            raise PipelineError(f"No input file configured for source '{source_key}'")
        raw = read_table(file_path, csv_cfg)
        canonical, source_warnings = canonicalize_source(raw, source_key, source_cfg, config["company"]["id"])
        canonical["source_config_key"] = source_key
        source_frames.append(canonical)
        warnings.extend(source_warnings)
        LOGGER.info("Loaded %s: %d rows from %s", source_key, len(canonical), file_path.name)

    if not source_frames:
        raise PipelineError("No source files are configured")
    records = pd.concat(source_frames, ignore_index=True)
    if records["record_id"].duplicated().any():
        duplicates = records.loc[records["record_id"].duplicated(), "record_id"].head(10).tolist()
        raise PipelineError(f"record_id values must be unique across sources; duplicates include {duplicates}")

    taxonomy_path = resolve_path(config_dir, files_cfg.get("taxonomy"))
    if taxonomy_path is None:
        raise PipelineError("files.taxonomy is required")
    taxonomy_columns = config["taxonomy"]["columns"]
    taxonomy, taxonomy_warnings = load_taxonomy(taxonomy_path, taxonomy_columns, csv_cfg)
    warnings.extend(taxonomy_warnings)

    # ---- Labels / memory ----
    labels_path = resolve_path(config_dir, files_cfg.get("labels"))
    label_config = dict(config.get("labels", {}))
    if disable_label_memory:
        label_config["enabled"] = False
    label_examples = pd.DataFrame()
    supplier_overrides: dict[tuple[str, str], str] = {}
    item_overrides: dict[tuple[str, str], str] = {}
    for column in ("label_l1", "label_l2", "label_l3", "label_l4", "label_supplier_key", "label_item_key"):
        records[column] = ""

    if attach_run_labels:
        answer_key_config = {**label_config, "enabled": True}
        records, label_examples, label_warnings = attach_labels(records, labels_path, answer_key_config, csv_cfg)
        warnings.extend(label_warnings)
    elif label_config.get("enabled") and label_config.get("memory_sources"):
        # Preferred: memory comes from the original labelled file, not from the rows being classified.
        label_examples, label_warnings = build_memory_examples(config, config_dir, records)
        warnings.extend(label_warnings)
    elif label_config.get("enabled"):
        warnings.append(
            "labels.enabled without labels.memory_sources attaches labels to the run rows themselves: "
            "accuracy on those rows is inflated. Use memory_sources for real runs."
        )
        records, label_examples, label_warnings = attach_labels(records, labels_path, label_config, csv_cfg)
        warnings.extend(label_warnings)
        if label_config.get("use_golden_keys", False):
            supplier_overrides, item_overrides, identity_warnings = label_identity_overrides(records)
            warnings.extend(identity_warnings)

    # Spend decisions precede currency conversion so rejected rows do not fail on irrelevant FX.
    records = apply_exclusions(records, source_configs, config.get("exclusions", {}))
    if not label_examples.empty and not label_config.get("memory_sources") and not attach_run_labels:
        eligible_reference_ids = set(records.loc[records["include"] == 1, "record_id"].astype(str))
        label_examples = label_examples.loc[label_examples["record_id"].astype(str).isin(eligible_reference_ids)].copy()

    fx_path = resolve_path(config_dir, files_cfg.get("fx_rates"))
    fx_df = read_table(fx_path, csv_cfg) if fx_path and fx_path.exists() else None
    if fx_path and not fx_path.exists():
        raise PipelineError(f"Configured FX file does not exist: {fx_path}")
    fx_lookup = build_fx_lookup(fx_df, config.get("fx"))
    records, amount_warnings = normalize_amounts(
        records,
        source_configs,
        str(config["company"]["reporting_currency"]).upper(),
        fx_lookup,
        config.get("cleaning", {}),
    )
    warnings.extend(amount_warnings)
    # Unconvertible rows are kept with spend=0 and reported; aborting would publish nothing at all.
    unconverted_rows = int(((records["include"] == 1) & records["spend_unconverted"]).sum())
    fx_approximated_rows = int(records.attrs.get("fx_approximated_rows", 0))

    reconciliation_metrics: dict[str, int] = {"matched": 0, "unmatched_tool": 0, "ambiguous": 0, "tool_export_duplicates": 0}
    reconciliation_cfg = config.get("reconciliation", {})
    matches: dict[str, str] = {}
    if reconciliation_cfg.get("enabled"):
        tool_source_cfg = source_configs.get(reconciliation_cfg.get("tool_source", "tool"), {})
        tool_name = str(tool_source_cfg.get("source_name", "TOOL"))
        erp_source_cfg = source_configs.get(reconciliation_cfg.get("erp_source", "erp"), {})
        erp_name = str(erp_source_cfg.get("source_name", "ERP"))
        if reconciliation_cfg.get("invoice_export_deduplication", True):
            records, duplicate_count = deduplicate_tool_exports(records, tool_name)
            reconciliation_metrics["tool_export_duplicates"] = duplicate_count
        records, matches, match_metrics = reconcile_erp_tool(
            records, reconciliation_cfg, erp_source=erp_name, tool_source=tool_name
        )
        reconciliation_metrics.update(match_metrics)

    # Entity keys come from the algorithm only. Golden keys are never copied in unless
    # labels.use_golden_keys is explicitly set (it is off by default because it fakes dedupe quality).
    records = assign_entity_keys(records, config.get("cleaning", {}).get("missing_supplier_policy", "unique_per_record"))
    if supplier_overrides or item_overrides:
        records = _apply_identity_overrides(records, supplier_overrides, item_overrides)
    if matches:
        records = inherit_matched_identity(records, matches)

    class_config = dict(config.get("classification", {}))
    if disable_label_memory:
        class_config["use_label_memory"] = False
    return {
        "records": records,
        "taxonomy": taxonomy,
        "taxonomy_columns": taxonomy_columns,
        "label_examples": label_examples,
        "class_config": class_config,
        "warnings": warnings,
        "reconciliation_metrics": reconciliation_metrics,
        "unconverted_rows": unconverted_rows,
        "fx_approximated_rows": fx_approximated_rows,
    }


def run_pipeline(
    config: dict[str, Any],
    config_dir: Path,
    output_dir: Path,
    candidate_name: str = "candidate",
    disable_label_memory: bool = False,
    output_tag: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = prepare_records(config, config_dir, disable_label_memory=disable_label_memory)
    records = prepared["records"]
    taxonomy = prepared["taxonomy"]
    taxonomy_columns = prepared["taxonomy_columns"]
    label_examples = prepared["label_examples"]
    class_config = prepared["class_config"]
    warnings = prepared["warnings"]

    # Classification uses only included records. Excluded rows remain in output with blank taxonomy.
    included_mask = records["include"] == 1
    included = records.loc[included_mask].copy()
    classified, class_warnings = classify_records(
        included,
        taxonomy,
        taxonomy_columns,
        label_examples if not disable_label_memory else None,
        class_config,
        config_dir,
    )
    warnings.extend(class_warnings)
    for depth in range(1, 5):
        records[f"l{depth}"] = ""
    records["confidence"] = 0.0
    if not classified.empty:
        for depth in range(1, 5):
            records.loc[classified.index, f"l{depth}"] = classified[f"l{depth}"].values
        records.loc[classified.index, "confidence"] = classified["confidence"].values

    final_output = _make_output(records, config)
    validation_metrics = validate_output(records, final_output, taxonomy, taxonomy_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    company_id = str(config["company"]["id"]).upper()
    file_tag = str(output_tag).strip().upper() if output_tag else company_id
    filename = (
        f"{candidate_name}__{file_tag}.csv"
        if output_tag
        else str(config.get("company", {}).get("output_filename", f"{candidate_name}__{company_id}.csv"))
    )
    filename = filename.replace("candidate", candidate_name)
    output_path = output_dir / filename
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    final_output.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(output_path)

    reason_counts = Counter(final_output.loc[final_output["include"] == 0, "exclusion_reason"].astype(str))
    source_counts = records.groupby("source").size().to_dict()
    report = {
        "company_id": company_id,
        "output_tag": file_tag,
        "output_file": str(output_path),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        **validation_metrics,
        "source_rows": {str(key): int(value) for key, value in source_counts.items()},
        "exclusion_reason_counts": dict(reason_counts),
        "date_parse_failures": int(records["date_parse_failed"].sum()),
        "fx_rows_nearest_month": prepared["fx_approximated_rows"],
        "included_rows_unconverted_spend_zero": prepared["unconverted_rows"],
        "reconciliation": prepared["reconciliation_metrics"],
        "label_reference_examples": int(len(label_examples)),
        "label_memory_enabled": bool(class_config.get("use_label_memory", True) and not disable_label_memory),
        "warnings": warnings[:200],
        "warning_count": len(warnings),
        "validation": "passed",
    }
    report_path = output_dir / f"{candidate_name}__{file_tag}__run_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote output: %s", output_path)
    LOGGER.info("Run report: %s", report_path)
    return report
