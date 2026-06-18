# Deal Opportunities

Python notebooks and supporting workbooks for deal analytics.

## Current Workstreams

- `Affirm/` - recent 2026-X1 analysis notebook and pricing outputs.
- `Project Driver Recreation Vehicles/` - current Recreation Vehicles workflow, baseline curves, CPR/CDR/DQ outputs, and final cash flow output.

## Shared Code

- `loanPipelineHelpers.py` - quiet reusable pipeline helpers for reading tabular files, standardizing columns, quality checks, loan/performance merges, panel shaping, target creation, and performance rollups.
- `parentHelpers.py` - notebook-facing helpers for display, formatting, plotting, Data Wrangler convenience loading, and shared exploratory deal-review tables.
- `regression.py` - current shared regression pipeline used by the recent notebooks.
- `helperProjectDriver.py` - Project Driver / Recreation Vehicles-specific curve, payment, prepayment, and cleanup helpers.

Rule of thumb for adding new helpers:

- Put reusable loan tape transformations in `loanPipelineHelpers.py`.
- Put notebook display, plotting, and presentation formatting in `parentHelpers.py`.
- Put model training or coefficient/diagnostic logic in `regression.py`.
- Put one-deal-only assumptions or cleanup in that deal's helper file.

## Legacy Candidates

These files are not imported by the recent Affirm or Project Driver notebooks, but are still left in place until their old deal dependencies are reviewed:

- `linearRegression.py`
- `regression_old.py`
- `logisticRegression.py`
- `Navitas/linear_regression.py`

## Notes

- Several workflows still read deal source files from external `F:\...` paths.
- Some files with an `.xlsx` extension are older Excel/OLE workbooks, so `openpyxl` may not be able to inspect them even if Excel can open them.
- Data files and generated workbooks are intentionally ignored by Git. See `DATA_ORGANIZATION.md` and `data_manifest.md` for the recommended local data layout and current workbook inventory.
