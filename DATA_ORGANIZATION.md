# Data Organization

This repo should be treated as code-first. Keep Excel workbooks, source tapes, generated outputs, PDFs, and local scratch files outside Git history.

## Recommended Data Root

Use a sibling project-data area, for example:

```text
Deal Opportunities Data/
  Affirm/
    00_raw/
    01_working/
    02_cleaned/
    03_outputs/
    99_archive/

  Project Driver Recreation Vehicles/
    00_raw/
    01_working/
    02_cleaned/
    03_outputs/
    99_archive/
```

## Folder Meanings

- `00_raw/`: original files exactly as received. Do not edit these.
- `01_working/`: manual Excel cleanup, temporary analysis, and reviewer workbooks.
- `02_cleaned/`: cleaned model-ready data files, preferably Parquet or CSV exports used by notebooks.
- `03_outputs/`: generated outputs, coefficient files, charts, cash-flow outputs, and review workbooks.
- `99_archive/`: old versions that are not active but may be useful for reference.

## Naming Convention

Prefer explicit dates and versions:

```text
project_driver_origination_asof_2026-03-31_received_2026-05-06.xlsx
project_driver_performance_asof_2026-03-31_received_2026-05-06.xlsx
project_driver_cleaned_panel_asof_2026-03-31_v01.parquet
project_driver_regression_coefficients_run_2026-05-06.xlsx
```

Avoid ambiguous names like `Latest`, `Final`, and `Final_v2` when creating new files.

## Current Compatibility Note

The current notebooks still use some relative workbook paths inside deal folders. To avoid breaking those notebooks, local workbook files can stay in place on disk, but they should remain ignored by Git.
