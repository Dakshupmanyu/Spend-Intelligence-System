"""Honest evaluation against the golden labels.

The golden labels are only an answer key here. A row is never classified with its own label,
and the held-out modes also keep every row of the same supplier (or item) out of the memory.

Modes
  zero-label        no label memory at all: classify from the category tree alone
  supplier-holdout  memory built from labelled rows of OTHER suppliers (k folds)
  item-holdout      memory built from labelled rows of OTHER items (k folds)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .classify import classify_records
from .memory import examples_from_labelled_records
from .pipeline import PipelineError, prepare_records

MODES = ("zero-label", "supplier-holdout", "item-holdout")


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def pairwise_precision_recall(predicted: pd.Series, gold: pd.Series) -> dict[str, float | int]:
    """Pair-counting precision/recall of a grouping against golden groups."""
    frame = pd.DataFrame({"p": predicted.astype(str).values, "g": gold.astype(str).values})
    true_pairs = sum(_pairs(int(c)) for c in frame.groupby(["p", "g"]).size())
    predicted_pairs = sum(_pairs(int(c)) for c in frame.groupby("p").size())
    gold_pairs = sum(_pairs(int(c)) for c in frame.groupby("g").size())
    return {
        "rows": int(len(frame)),
        "precision": round(true_pairs / predicted_pairs, 4) if predicted_pairs else 1.0,
        "recall": round(true_pairs / gold_pairs, 4) if gold_pairs else 1.0,
        "predicted_groups": int(frame["p"].nunique()),
        "golden_groups": int(frame["g"].nunique()),
    }


def _fold_of(group: str, folds: int) -> int:
    return int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % folds


def _level_metrics(frame: pd.DataFrame, levels: int) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    weight = frame["spend_pre_exclusion"].abs()
    for level in range(1, levels + 1):
        have_label = frame[f"label_l{level}"] != ""
        if not have_label.any():
            continue
        subset = frame.loc[have_label]
        # "path" accuracy: this level and every level above it must be right
        correct = pd.Series(True, index=subset.index)
        for depth in range(1, level + 1):
            correct &= subset[f"l{depth}"] == subset[f"label_l{depth}"]
        w = weight.loc[subset.index]
        metrics[f"level_{level}"] = {
            "rows": int(len(subset)),
            "path_accuracy": round(float(correct.mean()), 4),
            "spend_weighted_path_accuracy": round(float((w * correct).sum() / w.sum()), 4) if w.sum() > 0 else 0.0,
            "majority_class_baseline": round(float(subset[f"label_l{level}"].value_counts(normalize=True).iloc[0]), 4),
        }
    return metrics


def evaluate(
    config: dict[str, Any], config_dir: Path, mode: str = "zero-label", folds: int = 3, output_dir: Path | None = None
) -> dict[str, Any]:
    if mode not in MODES:
        raise PipelineError(f"mode must be one of {MODES}")
    prepared = prepare_records(config, config_dir, attach_run_labels=True)
    records, taxonomy, columns = prepared["records"], prepared["taxonomy"], prepared["taxonomy_columns"]
    class_config = prepared["class_config"]

    truth = records.loc[(records["include"] == 1) & (records["label_l1"] != "")].copy()
    if truth.empty:
        raise PipelineError("No included rows have golden labels; check labels config and join column")

    predictions: list[pd.DataFrame] = []
    if mode == "zero-label":
        classified, _ = classify_records(truth, taxonomy, columns, None, {**class_config, "use_label_memory": False}, config_dir)
        predictions.append(classified)
    else:
        group_column = "label_supplier_key" if mode == "supplier-holdout" else "label_item_key"
        groups = truth[group_column].where(truth[group_column] != "", truth["record_id"])
        truth["_fold"] = groups.map(lambda value: _fold_of(str(value), folds))
        for fold in range(folds):
            test = truth.loc[truth["_fold"] == fold]
            train = truth.loc[truth["_fold"] != fold]
            if test.empty:
                continue
            examples = examples_from_labelled_records(train)
            classified, _ = classify_records(
                test, taxonomy, columns, examples, {**class_config, "use_label_memory": True}, config_dir
            )
            predictions.append(classified)

    scored = pd.concat(predictions).sort_index()
    for column in ("l1", "l2", "l3", "l4"):
        scored[column] = scored[column].fillna("").astype(str)
    report: dict[str, Any] = {
        "company": str(config["company"]["id"]),
        "mode": mode,
        "folds": folds if mode != "zero-label" else None,
        "evaluated_rows": int(len(scored)),
        "classification": _level_metrics(scored, min(4, len(columns))),
        "dedupe_vs_golden_keys": {},
        "note": (
            "Dedupe metrics use the engine's own keys (golden keys are never copied in). "
            "Held-out modes keep all rows of the same supplier/item out of the label memory."
        ),
    }
    for name, predicted_col, gold_col in (
        ("supplier", "supplier_key", "label_supplier_key"),
        ("item", "item_key", "label_item_key"),
    ):
        usable = records.loc[(records["include"] == 1) & (records[gold_col] != "")]
        if not usable.empty:
            report["dedupe_vs_golden_keys"][name] = pairwise_precision_recall(usable[predicted_col], usable[gold_col])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"evaluate__{report['company']}__{mode}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
