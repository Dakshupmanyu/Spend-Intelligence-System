from pathlib import Path

import pandas as pd
import yaml

from spend_engine.config import load_config
from spend_engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def _run(config_name: str, tmp_path: Path, disable_label_memory: bool = True):
    config, config_dir = load_config(ROOT / "configs" / config_name)
    return run_pipeline(
        config,
        config_dir,
        tmp_path,
        candidate_name="pytest",
        disable_label_memory=disable_label_memory,
    )


def test_company_a_end_to_end_reconciles_and_preserves_all_records(tmp_path):
    report = _run("synthetic_a.yaml", tmp_path)
    output = pd.read_csv(tmp_path / "pytest__A.csv", keep_default_na=False)
    assert len(output) == 7
    assert output["record_id"].is_unique
    assert report["reconciliation"]["matched"] == 2
    assert report["reconciliation"]["tool_export_duplicates"] == 1
    assert (output["exclusion_reason"] == "DUPLICATE").sum() == 3
    assert (output["exclusion_reason"] == "NOT_APPROVED").sum() == 1
    assert (output.loc[output["include"] == 0, "spend"] == 0).all()
    assert set(output.loc[output["include"] == 1, "l1"]).issubset({"Mining Equipment", "Operating Materials"})


def test_company_b_end_to_end_fx_and_taxonomy(tmp_path):
    # Synthetic fixture uses an explicit document-type cancellation code; this is not asserted to be B's real code.
    config_path = ROOT / "configs" / "synthetic_b.yaml"
    config_data = yaml.safe_load(config_path.read_text())
    config_data["exclusions"]["cancelled_document_types"] = ["CAN"]
    temp_config = tmp_path / "synthetic_b_test.yaml"
    # Make fixture paths absolute so the temporary config remains portable within this test.
    for key, value in list(config_data["files"].items()):
        config_data["files"][key] = str((ROOT / "configs" / value).resolve())
    temp_config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    config, config_dir = load_config(temp_config)
    report = run_pipeline(config, config_dir, tmp_path / "out", candidate_name="pytest", disable_label_memory=True)
    output = pd.read_csv(tmp_path / "out" / "pytest__B.csv", keep_default_na=False)
    assert len(output) == 4
    eur_row = output.loc[output["record_id"] == "B000002"].iloc[0]
    assert round(float(eur_row["spend"]), 2) == 200.00
    monthly_fx_row = output.loc[output["record_id"] == "B000004"].iloc[0]
    assert round(float(monthly_fx_row["spend"]), 2) == 100.00
    assert output.loc[output["record_id"] == "B000003", "exclusion_reason"].iloc[0] == "CANCELLED"
    assert report["validation"] == "passed"


def test_a_strict_zero_label_mode_is_supported(tmp_path):
    report = _run("synthetic_a.yaml", tmp_path, disable_label_memory=True)
    assert report["label_memory_enabled"] is False
    assert report["validation"] == "passed"


def test_historical_label_memory_can_be_enabled_without_copying_labels(tmp_path):
    config, config_dir = load_config(ROOT / "configs" / "synthetic_a.yaml")
    report = run_pipeline(config, config_dir, tmp_path, candidate_name="memory", disable_label_memory=False)
    output = pd.read_csv(tmp_path / "memory__A.csv", keep_default_na=False)
    assert report["label_memory_enabled"] is True
    included = output.loc[output["include"] == 1]
    assert included[["l1", "l2", "l3"]].notna().all().all()
    assert set(included["l1"]).issubset({"Mining Equipment", "Operating Materials"})


def test_synthetic_company_c_works_with_unseen_four_level_taxonomy_and_no_labels(tmp_path):
    config, config_dir = load_config(ROOT / "configs" / "synthetic_c.yaml")
    report = run_pipeline(config, config_dir, tmp_path, candidate_name="pytest", disable_label_memory=False)
    output = pd.read_csv(tmp_path / "pytest__C.csv", keep_default_na=False)
    assert report["label_reference_examples"] == 0
    assert report["label_memory_enabled"] is False
    assert len(output) == 3
    assert output.loc[output["record_id"] == "C000003", "exclusion_reason"].iloc[0] == "CANCELLED"
    assert output.loc[output["include"] == 1, "l4"].astype(str).str.len().gt(0).all()


def test_output_tag_supports_blind_test_filenames_without_config_edit(tmp_path):
    config, config_dir = load_config(ROOT / "configs" / "synthetic_a.yaml")
    report = run_pipeline(
        config,
        config_dir,
        tmp_path,
        candidate_name="blind",
        disable_label_memory=True,
        output_tag="A2",
    )
    assert (tmp_path / "blind__A2.csv").is_file()
    assert (tmp_path / "blind__A2__run_report.json").is_file()
    assert report["output_tag"] == "A2"
