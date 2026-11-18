import pandas as pd

from spend_engine.reconcile import reconcile_erp_tool


def test_single_invoice_matches_unique_sum_of_multiple_receipts():
    frame = pd.DataFrame([
        {"record_id": "ERP:E1", "source": "ERP", "include": 1, "exclusion_reason": "", "po_number": "4500000001", "po_item": "10", "spend_pre_exclusion": 400.0, "supplier_name": "Acme Engineering", "description": "industrial pump", "supplier_key": "S1", "item_key": "I1"},
        {"record_id": "ERP:E2", "source": "ERP", "include": 1, "exclusion_reason": "", "po_number": "4500000001", "po_item": "10", "spend_pre_exclusion": 600.0, "supplier_name": "Acme Engineering", "description": "industrial pump", "supplier_key": "S1", "item_key": "I1"},
        {"record_id": "TOOL:T1", "source": "TOOL", "include": 1, "exclusion_reason": "", "po_number": "4500000001", "po_item": "10", "spend_pre_exclusion": 1000.0, "supplier_name": "ACME ENGINEERING", "description": "industrial pump", "supplier_key": "", "item_key": ""},
    ])
    result, matches, metrics = reconcile_erp_tool(
        frame,
        {"amount_absolute_tolerance": 2.0, "amount_relative_tolerance": 0.02, "require_unique_best_match": True, "min_supplier_similarity": 55, "min_description_similarity": 8},
    )
    assert matches == {"TOOL:T1": "ERP:E1"}
    assert metrics["multi_receipt_matches"] == 1
    assert metrics["matched_erp_records"] == 2
    assert result.loc[result["record_id"] == "TOOL:T1", "exclusion_reason"].iloc[0] == "DUPLICATE"
