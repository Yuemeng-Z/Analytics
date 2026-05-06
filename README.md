# Deal Opportunities

Python notebooks and supporting workbooks for deal analytics.

## Current Workstreams

- `Affirm/` - recent 2026-X1 analysis notebook and pricing outputs.
- `Project Driver Recreation Vehicles/` - current Recreation Vehicles workflow, baseline curves, CPR/CDR/DQ outputs, and final cash flow output.

## Shared Code

- `parentHelpers.py` - plotting, formatting, weighted-average, and vintage-analysis helpers used by the recent notebooks.
- `regression.py` - current shared regression pipeline used by the recent notebooks.

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
