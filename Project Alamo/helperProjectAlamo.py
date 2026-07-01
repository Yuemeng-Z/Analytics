"""
Project Alamo-specific cleanup helpers.

Keep Alamo-only source field cleanup and category normalization here. Generic
pipeline utilities belong in loanPipelineHelpers.py and shared plotting/display
helpers belong in parentHelpers.py.
"""

import pandas as pd


def _normalize_category(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("&", " and ", regex=False)
        .str.replace("/", " ", regex=False)
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )


def clean_land_type(
    df,
    source_col="land_type",
    output_col="land_type_clean",
    unknown_label="Unknown",
):
    """
    Standardize Project Alamo land type wording.

    Keeps source_col unchanged and writes cleaned values to output_col.
    """
    if source_col not in df.columns:
        raise ValueError(f"Missing land type column: {source_col}")

    land_type_map = {
        'community': 'Manufactured Home Community', 
        'customer owned private land': 'Customer Owned Private Land',
       'family land': 'Customer Owned Private Land',
       'family land not owner by customer': 'Private Land Not Owned By Customer',
       'family land- no rent': 'Customer Owned Private Land',
       'manufactured home community': 'Manufactured Home Community',
       'owned property': 'Customer Owned Private Land',
       'owned property ': 'Customer Owned Private Land',
       'owned property land contract/ mortgage trust deed': 'Customer Owned Private Land',
       'owned property w/ no lien': 'Customer Owned Private Land',
       'owned property w/ no lien': 'Customer Owned Private Land',
       'owned property with no lien': 'Customer Owned Private Land',
       'owned propety ': 'Customer Owned Private Land',
       'private land not owned by customer': 'Private Land Not Owned By Customer',
    }

    out = df.copy()
    normalized = _normalize_category(out[source_col])
    out[output_col] = normalized.map(land_type_map).fillna(unknown_label)
    out.loc[out[source_col].isna(), output_col] = pd.NA
    return out


def clean_borrower_id_type(
    df,
    source_col="borrower_id_type",
    output_col="borrower_id_type_clean",
    unknown_label="Unknown",
):
    """
    Standardize Project Alamo borrower ID type wording.

    Keeps source_col unchanged and writes cleaned values to output_col.
    """
    if source_col not in df.columns:
        raise ValueError(f"Missing borrower ID type column: {source_col}")

    borrower_id_type_map = {
        "driver license": "Driver License",
        "drivers license": "Driver License",
        "driver's license": "Driver License",
        "drivers licence": "Driver License",
        "driver licence": "Driver License",
        "dl": "Driver License",
        "d l": "Driver License",
        "passport": "Passport",
        "us passport": "Passport",
        "u s passport": "Passport",
        "state id": "State ID",
        "state identification": "State ID",
        "identification card": "State ID",
        "id card": "State ID",
        "military id": "Military ID",
        "military identification": "Military ID",
        "permanent resident card": "Permanent Resident Card",
        "green card": "Permanent Resident Card",
    }

    out = df.copy()
    normalized = _normalize_category(out[source_col]).str.replace("'", "", regex=False)
    out[output_col] = normalized.map(borrower_id_type_map).fillna(unknown_label)
    out.loc[out[source_col].isna(), output_col] = pd.NA
    return out

def clean_borrower_income(row):
    if row['borrower_income_frequency_clean'] == 'Monthly':
        return row['borrower_income'] * 12
    elif row['borrower_income_frequency_clean'] == 'Bi-Weekly':
        return row['borrower_income'] * 26
    elif row['borrower_income_frequency_clean'] == 'Weekly':
        return row['borrower_income'] * 52
    else:
        return row['borrower_income'] * 12
    
def clean_borrower_income_frequency(
    df,
    source_col="borrower_income_frequency",
    output_col="borrower_income_frequency_clean",
    unknown_label="Unknown",
):
    """
    Standardize Project Alamo borrower income frequency wording.

    Keeps source_col unchanged and writes cleaned values to output_col.
    """
    if source_col not in df.columns:
        raise ValueError(f"Missing borrower income frequency column: {source_col}")

    income_frequency_map = {
        "weekly": "Weekly",
        "week": "Weekly",
        "wkly": "Weekly",
        "bi weekly": "Bi-Weekly",
        "biweekly": "Bi-Weekly",
        "bi week": "Bi-Weekly",
        "BiWeek": "Bi-Weekly",
        "every 2 weeks": "Bi-Weekly",
        "every two weeks": "Bi-Weekly",
        "semi monthly": "Bi-Weekly",
        "semimonthly": "Bi-Weekly",
        "twice monthly": "Bi-Weekly",
        "twice a month": "Bi-Weekly",
        "monthly": "Monthly",
        "month": "Monthly",
        "mthly": "Monthly",
        "annual": "Annual",
        "annually": "Annual",
        "Annually": "Annual",
        "yearly": "Annual",
        "year": "Annual",
    }

    out = df.copy()
    normalized = _normalize_category(out[source_col])
    out[output_col] = normalized.map(income_frequency_map).fillna(unknown_label)
    out.loc[out[source_col].isna(), output_col] = pd.NA
    return out


def clean_alamo_categories(
    df,
    land_type_col="land_type",
    borrower_id_type_col="borrower_id_type",
    borrower_income_frequency_col="borrower_income_frequency",
):
    """
    Apply the standard Project Alamo categorical cleanups.
    """
    out = df.copy()

    if land_type_col in out.columns:
        out = clean_land_type(out, source_col=land_type_col)

    if borrower_id_type_col in out.columns:
        out = clean_borrower_id_type(out, source_col=borrower_id_type_col)

    if borrower_income_frequency_col in out.columns:
        out = clean_borrower_income_frequency(
            out,
            source_col=borrower_income_frequency_col,
        )

    return out


def category_cleanup_crosstab(df, original_col, clean_col):
    """
    Review how raw category values mapped to cleaned labels.
    """
    return pd.crosstab(df[original_col], df[clean_col], dropna=False)
