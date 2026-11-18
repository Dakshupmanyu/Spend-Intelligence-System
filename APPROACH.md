# APPROACH (complete with actual A/B runs before submission)

## Pipeline design

A single config-driven Python CLI ingests source files into a canonical transaction schema, normalizes dates/amounts/currencies, assigns exclusion reason codes, reconciles duplicate-system records, resolves supplier/item keys, classifies included lines against the runtime taxonomy, validates the output contract, and writes one CSV per company. Company-specific behavior is configuration; category names are never hard-coded.

## Matching and de-duplication

Company A uses normalized PO reference and PO item as candidate-generation keys, then compares VAT-adjusted net spend and supplier/description evidence. Ambiguous matches remain unmatched rather than being force-merged. Invoice-tool export duplicates are handled separately. Supplier IDs are namespace-scoped; A historical `supplier_key` / `item_key` labels can seed identity mappings where IDs agree. Tool rows that reconcile to ERP inherit the ERP keys. Company B vendor IDs are namespaced by source system, region and company code until a global crosswalk is proven.

## Classification

The category tree is loaded at runtime and traversed hierarchically. The optional `intfloat/multilingual-e5-small` embedding model supplies semantic similarity; token overlap provides lexical evidence for technical tokens, abbreviations and short descriptions. Historical A/B labels can be joined by exact record ID and used as reference examples, with the query's own example excluded. C disables label memory and uses only its unseen runtime taxonomy plus transaction fields. Similarity is not a calibrated probability.

## Leakage controls

Labels and golden entity keys are excluded from inference text. Exact ID joins are used only to build the historical reference store. The query record's own label embedding is removed from its reference prototype during scoring. For a defensible A zero-label benchmark, disable label memory and report a separate supplier/item-group holdout benchmark before claiming generalization.

## Results (fill from actual datasets)

| Measurement | Result |
|---|---|
| A label-hidden exact/parent accuracy | TODO |
| A supplier-held-out accuracy | TODO |
| A item-held-out accuracy | TODO |
| B held-out clear-cut accuracy | TODO |
| C synthetic unseen-tree accuracy, spend-weighted | TODO |
| Full A runtime | TODO |
| Full B runtime | TODO |
| 25,000-line runtime | TODO |
| Embedding API cost per 1,000 lines | $0 (local model; compute cost not included) |

The included synthetic tests are integration checks, not benchmark evidence.
