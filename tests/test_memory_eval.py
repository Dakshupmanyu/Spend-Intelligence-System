"""#8 historical label memory, #9 id reuse, #10 no golden keys, #11 honest evaluation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from spend_engine.config import load_config
from spend_engine.evaluate import evaluate, pairwise_precision_recall
from spend_engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
ERP_HEADER = "line_id,source_system,po_number,po_item,posting_date,material_number,material_short_text,service_short_text,value_zar,document_currency,vendor_id,supplier_name"
# Wording that the category names cannot explain: only a label memory can map it.
FANS = "ZORBAX GLIMBLE"
MESH = "QUILBO TRANSMOGRIFIER"


def _erp_rows(prefix: str, start: int, count: int, wording: str, supplier: str, currency: str = "ZAR"):
    return [
        f"{prefix}{start + i:06d},ERP-A1,45{start + i:08d},10,6/28/2024,MAT-{wording[:2]}{i},{wording} {100 + i},,100.00,{currency},VEND-{supplier}{i % 4},{supplier}{i % 4} PTY"
        for i in range(count)
    ]


def _project(tmp_path: Path, run_rows: list[str], train_rows: list[str], label_rows: list[str]) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("\n".join([ERP_HEADER, *train_rows]) + "\n", encoding="utf-8")
    (data / "run.csv").write_text("\n".join([ERP_HEADER, *run_rows]) + "\n", encoding="utf-8")
    (data / "tool.csv").write_text(
        "tool_record_id,invoice_number,invoice_date,supplier_name_as_captured,po_reference,line_text,quantity,gross_amount,vat_rate,currency,status\n",
        encoding="utf-8",
    )
    (data / "labels.csv").write_text(
        "record_id,category_l1,category_l2,category_l3,supplier_key,item_key\n" + "\n".join(label_rows) + "\n", encoding="utf-8"
    )
    (data / "taxonomy.csv").write_text(
        "category_l1,category_l2,category_l3\nMining Equipment,Electrical Equipment,Fans\nOperating Materials,Roof Support,Mesh\n",
        encoding="utf-8",
    )
    cfg = yaml.safe_load((ROOT / "configs" / "synthetic_a.yaml").read_text())
    cfg["files"] = {
        "erp": str(data / "run.csv"), "tool": str(data / "tool.csv"),
        "labels": str(data / "labels.csv"), "taxonomy": str(data / "taxonomy.csv"),
    }
    cfg["classification"] = {**cfg.get("classification", {}), "use_embeddings": False, "use_label_memory": True}
    cfg["embeddings"] = {"use_embeddings": False}
    cfg["labels"] = {
        "enabled": True, "join_column": "record_id",
        "category_columns": ["category_l1", "category_l2", "category_l3"],
        "supplier_key_column": "supplier_key", "item_key_column": "item_key",
        "memory_sources": [{"source": "erp", "file": str(data / "train.csv")}],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _labels(rows: list[str], l1l2l3: tuple[str, str, str], tag: str) -> list[str]:
    # golden supplier key follows the row's own vendor, so supplier-holdout has real groups to hold out
    return [f"ERP:{row.split(',')[0]},{','.join(l1l2l3)},SUP-GOLD-{row.split(',')[10]},ITEM-GOLD-{row.split(',')[0]}" for row in rows]


FAN_LABEL = ("Mining Equipment", "Electrical Equipment", "Fans")
MESH_LABEL = ("Operating Materials", "Roof Support", "Mesh")


def _training():
    fans = _erp_rows("A", 1, 8, FANS, "ALPHA")
    mesh = _erp_rows("A", 101, 8, MESH, "BETA")
    return fans + mesh, _labels(fans, FAN_LABEL, "ALPHA") + _labels(mesh, MESH_LABEL, "BETA")


def _run(cfg_path: Path, out: Path, **kwargs):
    config, config_dir = load_config(cfg_path)
    report = run_pipeline(config, config_dir, out, candidate_name="t", **kwargs)
    return report, pd.read_csv(out / "t__A.csv", keep_default_na=False).set_index("record_id")


def test_new_batch_is_classified_from_historical_memory(tmp_path):
    train, labels = _training()
    new_batch = _erp_rows("N", 1, 2, FANS, "GAMMA") + _erp_rows("N", 11, 2, MESH, "DELTA")  # new ids, new suppliers
    cfg = _project(tmp_path, new_batch, train, labels)
    report, out = _run(cfg, tmp_path / "out")
    assert report["label_reference_examples"] == 16  # memory came from the training file, not the run
    assert out.loc[["ERP:N000001", "ERP:N000002"], "l3"].tolist() == ["Fans", "Fans"]
    assert out.loc[["ERP:N000011", "ERP:N000012"], "l3"].tolist() == ["Mesh", "Mesh"]


def test_rows_in_the_training_file_never_see_their_own_label(tmp_path):
    train, labels = _training()
    cfg = _project(tmp_path, train, train, labels)  # run file == training file
    report, out = _run(cfg, tmp_path / "out")
    assert report["label_reference_examples"] == 0  # every example is the same record as a run row
    assert any("same record" in w for w in report["warnings"])


def test_golden_supplier_and_item_keys_are_never_copied_to_output(tmp_path):
    train, labels = _training()
    cfg = _project(tmp_path, train, train, labels)
    _, out = _run(cfg, tmp_path / "out")
    assert not out["supplier_key"].astype(str).str.contains("GOLD").any()
    assert not out["item_key"].astype(str).str.contains("GOLD").any()


def test_reused_record_id_with_different_content_keeps_old_example_and_attaches_nothing(tmp_path):
    train, labels = _training()
    reused = [f"A000001,ERP-A1,4599999999,10,6/28/2024,MAT-ZZ,{MESH} 999,,100.00,ZAR,VEND-GAMMA,GAMMA PTY"]  # id of a FANS row
    cfg = _project(tmp_path, reused, train, labels)
    report, out = _run(cfg, tmp_path / "out")
    assert report["label_reference_examples"] == 16  # old labelled example kept, it is a different record
    assert any("reused by different records" in w for w in report["warnings"])
    assert out.loc["ERP:A000001", "l3"] == "Mesh"  # classified from its own text, not from the old label of that id


def test_value_zar_is_never_converted_again_whatever_document_currency_says(tmp_path):
    train, labels = _training()
    rows = _erp_rows("N", 1, 1, FANS, "GAMMA", currency="MZN") + _erp_rows("N", 5, 1, FANS, "GAMMA", currency="")
    cfg_path = _project(tmp_path, rows, train, labels)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["sources"]["erp"]["amount_in_reporting_currency"] = True
    cfg["sources"]["erp"]["currency_required"] = False
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    report, out = _run(cfg_path, tmp_path / "out")
    assert out["spend"].tolist() == [100.0, 100.0]
    assert report["included_rows_unconverted_spend_zero"] == 0


def test_evaluate_modes_report_levels_and_dedupe_without_copying_labels(tmp_path):
    train, labels = _training()
    cfg_path = _project(tmp_path, train, train, labels)
    config, config_dir = load_config(cfg_path)
    zero = evaluate(config, config_dir, "zero-label", output_dir=tmp_path / "rep")
    held = evaluate(config, config_dir, "supplier-holdout", folds=2, output_dir=tmp_path / "rep")
    assert zero["evaluated_rows"] == held["evaluated_rows"] == 16
    assert set(held["classification"]) == {"level_1", "level_2", "level_3"}
    # memory from OTHER suppliers still teaches the wording; zero-label cannot know it
    assert held["classification"]["level_3"]["path_accuracy"] > zero["classification"]["level_3"]["path_accuracy"]
    assert "supplier" in held["dedupe_vs_golden_keys"]
    assert (tmp_path / "rep" / "evaluate__A__supplier-holdout.json").is_file()


def test_pairwise_precision_recall_definition():
    predicted = pd.Series(["a", "a", "b", "b"])
    gold = pd.Series(["x", "x", "x", "y"])
    result = pairwise_precision_recall(predicted, gold)
    assert result["precision"] == 0.5 and result["recall"] == round(1 / 3, 4)
