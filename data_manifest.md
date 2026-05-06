# Data Manifest

This manifest records local data/workbook files used by the notebooks without storing those files in Git.

## Current Local Files

| Deal | File | Current Local Path | Suggested Folder | Notes |
| --- | --- | --- | --- | --- |
| Affirm | `7.33.7.1 26-X1 Affirm Performance Dashboard 2025_12_31.xlsx` | `Affirm/` | `00_raw/` | Source performance dashboard. |
| Affirm | `7.33.7.2 26-X1 Affirm Pool Factor Dashboard Latest 2025_12_31.xlsx` | `Affirm/` | `00_raw/` | Source pool factor dashboard. Rename future copies to avoid `Latest`. |
| Affirm | `7.33.7.3 26-X1 Affirm Prepayment Dashboard 2025_12_31.xlsx` | `Affirm/` | `00_raw/` | Source prepayment dashboard. |
| Affirm | `Analytics.xlsx` | `Affirm/` | `01_working/` | Local analysis workbook. |
| Affirm | `EverD30_Analysis.xlsx` | `Affirm/` | `01_working/` | Local analysis input/output. |
| Affirm | `FICO_Analysis.xlsx` | `Affirm/` | `01_working/` | Local FICO analysis workbook. |
| Affirm | `pricing_cdr_df_base_case_all.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Affirm | `pricing_cdr_df_base_case_C24.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Affirm | `pricing_cdr_df_worst_case.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Affirm | `pricing_cdrs.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Affirm | `pricing_cpr_df_base_case_all.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Affirm | `pricing_cpr_df_base_case_all_v2.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Affirm | `pricing_dq_df_base_case_all.xlsx` | `Affirm/` | `03_outputs/` | Generated pricing output. |
| Project Driver Recreation Vehicles | `Baseline_Inputs.xlsx` | `Project Driver Recreation Vehicles/` | `00_raw/` | Baseline curve input used by notebook. |
| Project Driver Recreation Vehicles | `Output_Final_v2.xlsx` | `Project Driver Recreation Vehicles/` | `01_working/` | Cash-flow workbook read by notebook. Rename future copies to avoid `v2` ambiguity. |
| Project Driver Recreation Vehicles | `CDRs.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Generated curve output. |
| Project Driver Recreation Vehicles | `CPRs.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Generated curve output. |
| Project Driver Recreation Vehicles | `DQ30.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Generated delinquency output. |
| Project Driver Recreation Vehicles | `cgl_by_quarter recfi.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Generated CGL output. |
| Project Driver Recreation Vehicles | `cnl_by_quarter recfi.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Generated CNL output. |
| Project Driver Recreation Vehicles | `Coefficient_output.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Regression output. |
| Project Driver Recreation Vehicles | `Coefficient_output_2.xlsx` | `Project Driver Recreation Vehicles/` | `03_outputs/` | Regression output. |

## Update Rule

When a notebook depends on a new external file, add a row here with the deal, filename, location, as-of or received date when known, and how the file is used.
