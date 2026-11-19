# DECISIONS AND ASSUMPTIONS

This document records decisions made from the task brief and pasted sample schemas. Validate each `TODO` against the full A/B data before the build freeze.

## Fixed by the task brief

- A ERP goods-receipt values are already net of tax.
- A invoice-tool amounts are gross; remove VAT using the row VAT rate, defaulting to 15% when blank.
- B and C report in USD. B `currency_rate` and the monthly FX file are units of local currency per USD; divide local-currency amounts by the rate.
- Count ERP records when an A purchase appears in both systems; mark the corresponding tool record `DUPLICATE`.
- Count an invoice-tool export only once.
- Only approved or paid invoices count.
- Output one row per input record, including excluded rows, with the specified reason codes and runtime taxonomy labels.
- C has no labels and an unseen taxonomy; no C label memory is permitted.

## Implemented baseline assumptions

- Exact record-ID joins are used for A/B labels. Fuzzy label joins are prohibited.
- A PO references such as `PO#4510000018/10.0` are parsed into PO number and item number.
- A reconciliation is conservative: PO/item candidates must also have close net amount and supporting supplier or description evidence. A unique subset-sum check can match one invoice to multiple receipt rows for the same PO item. Ties/complex groups remain unmatched.
- B vendor IDs are scoped by source system, region and company code to avoid merging IDs that may be local to each ERP.
- A missing supplier or item identifier produces a deterministic key from available name/description; when no evidence exists, the key is unique to the record rather than incorrectly grouping unknowns.
- A missing status field on ERP goods receipts does not block the receipt rows. A's invoice-tool status is required and only approved/paid is accepted.
- B has no status column in the supplied sample schema. A populated `paid_date` is interpreted as paid; when it is blank, the current config keeps the row with a report warning. This needs validation because the task's general rule says only approved or paid lines count.
- Generic `tax` and `duty` keywords are not used as blanket exclusions because they can describe legitimate procurement (e.g. customs duties).
- Unknown cancellation/document-type/account codes are not invented. Add them only after profiling the original files.
- Embedding similarity is combined with lexical evidence. It is not interpreted as a calibrated probability.

## Must validate against full A/B data

1. Are A PO numbers stable after normalization? What percentage of tool lines have PO references, parsed item numbers and net-amount matches?
2. Are there invoice-to-multiple-receipt or partial-receipt patterns? Does matching need group-level amount reconciliation?
3. What invoice-tool duplicate signatures are safe? Can repeated invoice lines share supplier, amount and description legitimately?
4. Which supplier and item IDs are global versus source/ERP scoped?
5. Which B document types, GL accounts or descriptions identify cancelled, reversed, tax, intercompany, royalty, revenue or financing records?
6. How should B rows with blank `paid_date` be interpreted when no approval-status field exists?
7. Which number/date formats, currencies, invalid FX months and malformed dates actually occur?
8. What are category accuracy and supplier/item clustering precision/recall on group-held-out data?

## Blind-test controls

- Do not commit original data or embedding-model binaries.
- Freeze Python code before the announced blind test.
- For C, create a YAML config from the profiler output and manually verify each mapping.
- Turn off labels and company-specific examples for C.
- Validate CSV row accounting, spend conversion, reason codes and taxonomy membership before publishing.

## Identity-resolution thresholds in the scaffold

- Supplier names without IDs first undergo case, punctuation, accent and legal-suffix normalization. Fuzzy merging is restricted to mutual-nearest name pairs scoring at least 97/100. An id-less name may inherit a scoped vendor ID only when the exact or high-confidence name match resolves to one unambiguous ID.
- Item IDs are authoritative only inside their configured source namespace. Description-only items with generic/low-information descriptions (for example `Product` or `Parts`) stay unique per record. Fuzzy description grouping requires a very high similarity score and identical numeric/technical tokens so different sizes or model numbers are not merged casually.
- These thresholds are conservative starting points, not empirically optimized values. Use the supplied labels to measure supplier/item pair precision and recall before the build freeze.

## Blind-test output naming

- The frozen CLI supports `--output-tag A2` and `--output-tag B2` so the same A/B configurations can emit the required blind-test filenames without a Python change. `--output-tag C` is unnecessary because the C config already emits the C suffix.

## Crash-safety decisions (patch 1)

- Missing FX month: use the nearest available month for that currency (earlier month wins a tie); no usable date uses the currency's latest month. Count shown in the run report as `fx_rows_nearest_month`.
- Currency absent from the FX table, blank/invalid amount, or unsupported reporting-currency conversion: the row stays included with spend 0 and is counted in `included_rows_unconverted_spend_zero`. Review these before trusting totals; they are not silently dropped.
- B FX month is taken from `posting_date` (fallback `invoice_date`). `invoice_date` can be years before the FX table (e.g. 2018).
- Malformed CSV lines are skipped and counted; a non-UTF-8 file is retried as cp1252 then latin-1. Blank-ID rows and repeated header rows are dropped with a warning (no ID is ever invented).
- Embedding model: cache first, then one-time download, then lexical fallback with a warning.

## Reconciliation decisions (patch 2)

- Ties between interchangeable ERP receipts (same supplier, text, amount) are not ambiguous: the tool invoice matches the lowest record id and each receipt is consumed once.
- A tie between genuinely different ERP rows still marks the tool row DUPLICATE (`ambiguous_policy: match_first`, default) because the purchase total is correct either way; identity keys are not inherited in that case. `ambiguous_policy: skip` restores the old behaviour, which double counts.
- Tool rows with no PO, or a PO the ERP never saw, use a fallback: normalised supplier similarity >= 85, net amount within max(1.00, 0.5%), description similarity >= 30, invoice date within 365 days of posting date. Supplier AND text are both required. A tool row whose PO exists in the ERP but fits nothing is left as tool-only spend. All thresholds are config keys (`fallback_*`) and are untuned: measure on the real A data.
- Report fields: `po_matches`, `multi_receipt_matches`, `fallback_matches`, `ambiguous_matched`, `ambiguous`.

## Identity-resolution performance (patch 3)

- Supplier/item fuzzy matching no longer compares all pairs. Names are blocked (shared 6-char prefix, 6-char suffix, whole token, or token head+tail; item blocks also require identical technical tokens), then scored with `rapidfuzz.process.cdist`. Oversized generic blocks use a sorted-neighbourhood window (`BLOCK_CAP`, `SORTED_WINDOW` in `identity.py`).
- Same decision rules as before (top match >= threshold, 3 points clear of runner-up, mutual nearest). Tests confirm identical clustering to brute force at the production thresholds (97/94/98). At much lower thresholds (e.g. 90) blocking can miss a few pairs, so do not lower the thresholds without re-checking.
- Measured on synthetic id-less data: 1,000 rows 9.6 s -> 0.2 s; 25,000 rows ~90 min (extrapolated) -> 6 s. A full 25,000-row run in lexical mode took 17 s end to end. Embedding time is not included: measure on the real laptop with the model loaded.

## Label memory, golden keys, evaluation (patch 4)

- Label memory comes from the original labelled file (`labels.memory_sources`), not from the rows being classified. A new batch (A2/B2) is classified with memory built from the old file; the new rows never receive labels.
- A run row never sees its own label: a memory example is dropped when its record id AND content fingerprint (text + supplier + raw amount) match a run row. Same id with different content means an id was reused by a new record: the old example is kept (it is still a correct example of the old record) and no label is attached to the new row. Consequence: running on the training file itself gives zero memory (honest zero-shot output for the A/B build files).
- Golden `supplier_key` / `item_key` are never copied into output (`labels.use_golden_keys: false`, the default). Dedupe quality is measured by `evaluate`, not faked. The dead keys `use_as_reference_examples` / `use_for_output_directly` were removed.
- `A ERP value_zar` is already net ZAR: `amount_in_reporting_currency: true` skips currency handling for it (before, a blank or foreign `document_currency` produced zero spend or a second conversion).
- `evaluate` command: `python -m spend_engine evaluate --config configs/company_a.yaml --mode zero-label|supplier-holdout|item-holdout [--folds 3]`. Reports path accuracy per level (plain and spend-weighted, with a majority-class baseline) and pairwise precision/recall of the engine's supplier/item keys against the golden keys. Holdout modes keep every row of a held-out supplier/item out of the memory.
