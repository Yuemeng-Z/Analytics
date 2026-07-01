# Deal Opportunities

Python notebooks and supporting workbooks for deal analytics.

## Current Workstreams

- `Affirm/` - recent analysis notebook and pricing outputs.
- `PNC Student Loans/` - private student loan processing and analysis notebook.
- `Project Alamo/` - cascade and CUSIP workflows with Alamo-specific helpers and SQL.
- `Project Driver Recreation Vehicles/` - recreation vehicles workflow, baseline curves, CPR/CDR/DQ outputs, and final cash flow output.
- `Project Sunrise/` - database connectivity and reconciliation SQL notebook.
- `Project Swift Lloyds/` - model development notebooks, SQL, and Swift-specific helper logic.
- `Revl/` - Revl processing notebook and outputs.

## Shared Code

- `loanPipelineHelpers.py` - quiet reusable pipeline helpers for reading tabular files, standardizing columns, quality checks, loan/performance merges, panel shaping, target creation, and performance rollups.
- `parentHelpers.py` - notebook-facing helpers for display, formatting, plotting, Data Wrangler convenience loading, and shared exploratory deal-review tables.
- `regression.py` - current shared regression pipeline used by the recent notebooks.
- `Project Driver Recreation Vehicles/helperProjectDriver.py` - Project Driver / Recreation Vehicles-specific curve, payment, prepayment, and cleanup helpers.
- `Project Alamo/helperProjectAlamo.py` - Alamo-specific helper functions.
- `Project Swift Lloyds/helperProjectSwift.py` - Swift/Lloyds-specific helper functions.

Rule of thumb for adding new helpers:

- Put reusable loan tape transformations in `loanPipelineHelpers.py`.
- Put notebook display, plotting, and presentation formatting in `parentHelpers.py`.
- Put model training or coefficient/diagnostic logic in `regression.py`.
- Put one-deal-only assumptions or cleanup in that deal's helper file.

## Notes

- Several workflows still read deal source files from external `F:\...` paths.
- Some files with an `.xlsx` extension are older Excel/OLE workbooks, so `openpyxl` may not be able to inspect them even if Excel can open them.
- Data files and generated workbooks are intentionally ignored by Git. See `DATA_ORGANIZATION.md` and `data_manifest.md` for the recommended local data layout and current workbook inventory.
