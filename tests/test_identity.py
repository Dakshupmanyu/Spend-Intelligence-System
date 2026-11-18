import pandas as pd

from spend_engine.identity import assign_entity_keys


def test_vendor_item_ids_are_namespaced_by_erp():
    rows = pd.DataFrame([
        {"record_id": "B1", "source": "B", "source_namespace": "ERP-1|R1|CO1", "item_namespace": "ERP-1|R1|CO1", "supplier_id": "V100", "supplier_name": "", "item_id": "M100", "description": "Bearing"},
        {"record_id": "B2", "source": "B", "source_namespace": "ERP-2|R2|CO2", "item_namespace": "ERP-2|R2|CO2", "supplier_id": "V100", "supplier_name": "", "item_id": "M100", "description": "Bearing"},
    ])
    result = assign_entity_keys(rows)
    assert result.loc[0, "supplier_key"] != result.loc[1, "supplier_key"]
    assert result.loc[0, "item_key"] != result.loc[1, "item_key"]


def test_name_only_supplier_aliases_merge_only_on_mutual_high_similarity():
    rows = pd.DataFrame([
        {"record_id": "1", "source": "C", "source_namespace": "C", "item_namespace": "C", "supplier_id": "", "supplier_name": "Acme Industrial Supply", "item_id": "I1", "description": "Hydraulic Valve 2 Inch"},
        {"record_id": "2", "source": "C", "source_namespace": "C", "item_namespace": "C", "supplier_id": "", "supplier_name": "Acme Industral Supply", "item_id": "I2", "description": "Hydraulic Valve 2 Inch"},
    ])
    result = assign_entity_keys(rows)
    assert result.loc[0, "supplier_key"] == result.loc[1, "supplier_key"]


def test_low_information_items_and_different_technical_sizes_do_not_merge():
    rows = pd.DataFrame([
        {"record_id": "1", "source": "C", "source_namespace": "C", "item_namespace": "C", "supplier_id": "", "supplier_name": "", "item_id": "", "description": "Product"},
        {"record_id": "2", "source": "C", "source_namespace": "C", "item_namespace": "C", "supplier_id": "", "supplier_name": "", "item_id": "", "description": "Product"},
        {"record_id": "3", "source": "C", "source_namespace": "C", "item_namespace": "C", "supplier_id": "", "supplier_name": "", "item_id": "", "description": "Hydraulic Pump 2 Inch"},
        {"record_id": "4", "source": "C", "source_namespace": "C", "item_namespace": "C", "supplier_id": "", "supplier_name": "", "item_id": "", "description": "Hydraulic Pump 3 Inch"},
    ])
    result = assign_entity_keys(rows)
    assert result.loc[0, "item_key"] != result.loc[1, "item_key"]
    assert result.loc[2, "item_key"] != result.loc[3, "item_key"]
