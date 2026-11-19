# Spend Intelligence Engine

A config-driven Python scaffold for the Cantill.ai spend-data task. One engine handles Company A, Company B and a newly configured Company C; company-specific mappings and taxonomies live in YAML/CSV files, not hard-coded category names.

> **Important:** This repository is a tested scaffold built from the supplied task brief and sample schemas. It has not been run against the confidential full-size A/B data. The A/B rules that require data discovery (notably cancellation/document-type codes, non-procurement accounts, PO reconciliation tolerances and vendor-ID namespaces) must be validated against those files before the build freeze.

## 1. What is included

- Config-driven CSV ingestion with UTF-8 BOM and delimiter detection.
- Canonical record representation so A, B and C share the same processing path.
- Amount parsing, gross-to-net VAT handling, row-level FX and monthly FX fallback.
- Explicit status and non-procurement exclusions with the task's four reason codes.
- Company A PO/item matching, unique multi-receipt subset reconciliation and duplicate invoice-export detection.
- Deterministic supplier/item keys, with A label keys used only as historical identity references.
- Runtime taxonomy loading for three- or four-level trees.
- Optional local ONNX embeddings through FastEmbed (`intfloat/multilingual-e5-small`), with an explicitly configured lexical-only mode.
- Exact-ID label joins and leave-one-out label-reference classification; no direct target-column copying to output.
- Output schema validation, atomic CSV publishing and JSON run reports.
- Synthetic A/B fixture data, integration tests, a schema profiler and a model-cache helper.

## 2. Repository layout

```text
spend-intelligence-engine/
├── configs/
│   ├── company_a.yaml
│   ├── company_b.yaml
│   ├── company_c.template.yaml
│   ├── synthetic_a.yaml
│   └── synthetic_b.yaml
├── data/                         # Create locally; confidential data is git-ignored
│   ├── company_a/
│   └── company_b/
├── fixtures/                     # Small synthetic tests only
├── scripts/
│   ├── cache_embedding_model.py
│   └── profile_input.py
├── src/spend_engine/              # Reusable engine
├── tests/
├── outputs/                       # Generated company CSVs
├── reports/                       # JSON run reports and profiling output
├── APPROACH.md
├── DECISIONS.md
└── pyproject.toml
```

## 3. Setup (Windows PowerShell)

Use Python 3.11 or 3.12. The project deliberately avoids a GPU requirement.

```powershell
cd spend-intelligence-engine
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[embeddings,dev]"
```

Download and warm the small multilingual embedding model **before the freeze**:

```powershell
python scripts/cache_embedding_model.py --model intfloat/multilingual-e5-small --cache-dir .cache_models
```

The model is cached locally. FastEmbed runs the ONNX model on CPU (`cuda=False`); no GPU is required. The scaffold checks the installed FastEmbed registry and registers `intfloat/multilingual-e5-small` as a custom model if that release does not list it by default, using FastEmbed's documented custom-model API ([FastEmbed README](https://github.com/qdrant/fastembed#dense-text-embeddings)). The model card specifies 384-dimensional embeddings, 100 supported languages, and the `query:` / `passage:` prefixes used by the configs ([model card](https://huggingface.co/intfloat/multilingual-e5-small)).

Do not commit `.cache_models/` or model binaries to the public repository. The engine tries the local cache first, then downloads the model once if the cache is empty. If the model still cannot load (no internet, no `fastembed`), it falls back to lexical matching and records a warning in the run report instead of aborting; set `fallback_to_lexical: false` in a config to make that a hard failure. Check the report for `Classifier ran in lexical-only fallback mode` before trusting a blind run. The synthetic test configs deliberately disable embeddings. Before freezing, run the cache helper and confirm it prints a vector dimension of 384 on the same laptop/environment that will execute the blind test.

For a no-embedding smoke test, install only the base and development dependencies and set `classification.use_embeddings: false` in a test config.

## 4. Add the actual company files locally

Create these folders and copy the confidential task files there. Do not commit them.

```text
 data/company_a/
   erp_goods_receipts.csv
   invoice_tool_export.csv
   labels_A.csv
   taxonomy_A.csv
 data/company_b/
   invoice_lines.csv
   fx_rates_usd.csv
   seed_labels.csv
   taxonomy_B.csv
```

The provided A/B YAML files point to these paths. If the supplied filenames differ, change only the `files` entries in YAML.

## 5. Run one company

```powershell
spend-engine run --config configs/company_a.yaml --output-dir outputs --candidate-name YOURNAME
spend-engine run --config configs/company_b.yaml --output-dir outputs --candidate-name YOURNAME
```

Equivalent module invocation:

```powershell
python -m spend_engine.cli run --config configs/company_a.yaml --output-dir outputs --candidate-name YOURNAME
```

At the blind test, the A2/B2 files use the same layouts. Place each new blind file at the configured local path (or update only the YAML `files` path) and run the frozen code with an output tag:

```powershell
spend-engine run --config configs/company_a.yaml --output-dir submission --candidate-name YOURNAME --output-tag A2
spend-engine run --config configs/company_b.yaml --output-dir submission --candidate-name YOURNAME --output-tag B2
```

This creates `YOURNAME__A2.csv` and `YOURNAME__B2.csv` without editing Python or changing the normal company mappings. For C, run the C config without an output tag to create `YOURNAME__C.csv`.

For the strict no-label A benchmark:

```powershell
spend-engine run --config configs/company_a.yaml --output-dir reports/a_zero_label --candidate-name YOURNAME --disable-label-memory
```

Expected output names:

- `outputs/YOURNAME__A.csv`
- `outputs/YOURNAME__B.csv`
- corresponding `__run_report.json` files

The CSV contains exactly:

```text
record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence
```

Every input record should appear once. Excluded rows have spend `0` and one of `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, or `DUPLICATE`.

## 6. Configure Company C without changing Python

1. Put the blind-test invoice-line and taxonomy files under `data/company_c/` (do not commit them).
2. Run the profiler:

   ```powershell
   python scripts/profile_input.py data/company_c/invoice_lines.csv --output reports/company_c_profile.json
   ```

3. Copy `configs/company_c.template.yaml` to `configs/company_c.yaml`.
4. Map actual column names and formats from the C files. Review the profile's suggestions manually; they are not trusted mappings.
5. Set the taxonomy columns in the exact L1-to-L4 order. If the tree has only three levels, use only three columns.
6. Map the actual C columns. For an input already denominated in USD with no currency column, set `currency_column: null`, `currency_required: false`, and `default_currency: USD`. For optional columns genuinely absent from C, set them to `null` and remove placeholder values from description lists.
7. Keep `labels.enabled: false` and `classification.use_label_memory: false` for C.
8. Run:

   ```powershell
   spend-engine run --config configs/company_c.yaml --output-dir submission --candidate-name YOURNAME
   ```

No Python code should be changed after the freeze. Validate every guessed column mapping against the actual header and a few representative values before the C run.

## 7. Tests

```powershell
pytest -q
```

The included fixtures are synthetic and small. They check VAT netting, A's example PO match, B's monthly FX conversion, taxonomy output, output row accounting and the no-label mode. Passing them does **not** prove performance on the full confidential data or unseen C data.

## 8. Operational safeguards

- Missing record IDs, amount columns, taxonomy columns, invalid FX for included rows, duplicate output IDs or out-of-tree predictions stop publication.
- Outputs are written to a temporary file and renamed only after validation passes.
- Input files are read as strings first so IDs do not lose leading zeroes.
- Category names are read at runtime; there are no industry-specific category literals in the classifier.
- No labels are directly copied into the output. Reference examples use exact ID joins and exclude the query record itself from its own prototype.
- The JSON report contains row counts, spend, exclusions, reconciliation counts, invalid-date counts and warnings.

## 9. Known limitations to validate before freeze

1. A reconciliation uses conservative one-to-one matching plus a unique bounded subset-sum check for invoices covering multiple receipts of the same PO item. It still does not prove every split/partial invoice pattern; inspect actual A distributions and review ambiguous groups.
2. Fuzzy supplier/item clustering is intentionally conservative in this first scaffold. The key generation is deterministic, and matched A tool rows inherit ERP identity, but ambiguous aliases should not be force-merged.
3. Company B's sample layout has no explicit approval-status column. A non-empty `paid_date` is interpreted as paid; blank `paid_date` remains eligible under the explicit `include_with_warning` configuration. Confirm this assumption from the task data/business context before final submission.
4. Actual cancellation document types, credit-note handling, GL/account exclusions and non-procurement rules are not inferable from the small pasted sample. Add validated codes to the YAML, not Python.
5. Embedding similarity is a ranking signal, not a calibrated probability. Confidence must be benchmarked and calibrated on held-out labels before presenting it as probability.
6. A2/B2 performance and the unseen C weighted-by-spend accuracy have not been established by the synthetic tests.

This scaffold is a start-to-run engineering baseline, not a claim that all hidden business rules have already been discovered.

## Measuring accuracy without leakage

```
python -m spend_engine evaluate --config configs/company_a.yaml --mode zero-label
python -m spend_engine evaluate --config configs/company_a.yaml --mode supplier-holdout --folds 3
python -m spend_engine evaluate --config configs/company_a.yaml --mode item-holdout --folds 3
```

`zero-label` is the number to quote for a company with no labels (like C). Reports are written to `reports/`.

## Finding what is not spend

```
python -m spend_engine suggest-exclusions --config configs/company_c.yaml
```

Ranks document types, account codes and suppliers by how much they look like tax, intercompany, royalty, financing, payroll or adjustments (or are mostly negative / blank). Read `reports/exclusion_suggestions__<company>.csv`, then add the values you trust to `exclusions.document_type_exclusions`, `exclusions.account_code_exclusions` or `exclusions.non_procurement_keywords`.
