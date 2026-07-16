"""
Shared notebook presentation and legacy cross-deal analysis helpers.

Put reusable table formatting, plotting, notebook display wrappers, and
exploratory deal-review helpers here. Core pipeline transformations belong in
loanPipelineHelpers.py, model training belongs in regression.py, and
deal-specific quirks belong in a deal helper module.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

from pathlib import Path
from loanPipelineHelpers import read_tabular, trim_last_n
from IPython.display import display

from scipy.stats import gaussian_kde
import seaborn as sns 

# ---------- Formatters ----------

DEFAULT_FICO_BUCKET_BINS = [-np.inf, 580, 620, 660, 700, 740, 780, 820, np.inf]
DEFAULT_FICO_BUCKET_LABELS = [
    "<580",
    "580-619",
    "620-659",
    "660-699",
    "700-739",
    "740-779",
    "780-819",
    "820+",
]


# NOTE:
# AI_PAYSTATUS_PRE_DEAL_ISSUANCE is treated as perfect pay only when the value is
# non-empty and every character in the pre-deal pay-status string is exactly "C".
def is_perfect_pay(paystatus):
    """
    Return True when a loan's pre-deal pay status is all "C".
    """
    if pd.isna(paystatus):
        return False

    paystatus = str(paystatus).strip()

    return len(paystatus) > 0 and all(status == "C" for status in paystatus)


def add_fico_bucket_column(
    df,
    fico_col="fico_score",
    bucket_col="fico_bucket",
    bins=None,
    labels=None,
    missing_label="Missing",
):
    """
    Add one categorical FICO bucket column for notebook analysis and plotting.
    """
    if fico_col not in df.columns:
        raise ValueError(f"Missing FICO column: {fico_col}")

    out = df.copy()
    bins = DEFAULT_FICO_BUCKET_BINS if bins is None else bins
    labels = DEFAULT_FICO_BUCKET_LABELS if labels is None else labels

    fico_numeric = pd.to_numeric(out[fico_col], errors="coerce")

    bucket = pd.cut(
        fico_numeric,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    out[bucket_col] = bucket.cat.add_categories([missing_label]).fillna(missing_label)
    return out

def fmt_percent(x, decimals=2):
    return "" if pd.isna(x) else f"{x:.{decimals}%}"

def fmt_millions(x, decimals=1):
    return "" if pd.isna(x) else f"{x/1_000_000:.{decimals}f}M"

def fmt_thousands(x, decimals=1):
    return "" if pd.isna(x) else f"{x/1_000:.{decimals}f}K"

def fmt_currency(x, decimals=2, symbol="$"):
    return "" if pd.isna(x) else f"{symbol}{x:,.{decimals}f}"

def fmt_accounting(x, decimals=2, symbol=""):
    if pd.isna(x):
        return ""
    return f"({symbol}{abs(x):,.{decimals}f})" if x < 0 else f"{symbol}{x:,.{decimals}f}"

def fmt_number(x, decimals=2):
    return "" if pd.isna(x) else f"{x:,.{decimals}f}"

# ---------- Main formatter ----------

def format_table(df, format_config):
    """
    format_config example:
    {
        "percent": ["margin", "growth"],
        "millions": ["revenue"],
        "accounting": ["profit"],
        "number": ["users"],
    }

    Optional: pass tuple for decimals:
        "percent": (["margin"], 1)
    """
    df_formatted = df.copy()

    formatter_map = {
        "percent": fmt_percent,
        "millions": fmt_millions,
        "thousands": fmt_thousands,
        "currency": fmt_currency,
        "accounting": fmt_accounting,
        "number": fmt_number,
    }

    for fmt_type, cols in format_config.items():
        if fmt_type not in formatter_map:
            continue

        # Handle optional decimals
        if isinstance(cols, tuple):
            col_list, decimals = cols
        else:
            col_list, decimals = cols, 2

        formatter = formatter_map[fmt_type]

        for col in col_list:
            if col not in df_formatted.columns:
                continue  # skip silently

            df_formatted[col] = df_formatted[col].apply(
                lambda x: formatter(x, decimals)
            )

    return df_formatted


# NOTE:
# Use regular display(df) / display(df.head()) for normal analysis and Data Wrangler work.
# That keeps numeric columns as real numbers, so notebook display right-aligns them naturally.
# auto_format_table is only for presentation/review tables where numbers are converted to
# strings like "1,234.00", "8.50%", or "(1,234.00)".
def auto_format_table(
    df,
    currency_cols=None,
    accounting_cols=None,
    percent_cols=None,
    number_cols=None,
    date_cols=None,
    exclude_cols=None,
    display_table=False,
    return_styler=False,
    percent_scale="auto",
    currency_decimals=2,
    accounting_decimals=2,
    percent_decimals=2,
    number_decimals=0,
    date_format="%Y-%m-%d",
):
    """
    Automatically format common finance table columns for notebook display.

    This does not change the source DataFrame.
    By default, it returns the formatted DataFrame copy for the normal notebook display.
    Pass return_styler=True to return a pandas Styler with right-aligned formatted numbers.
    Use explicit *_cols arguments to override or supplement the name-based guesses.

    percent_scale:
    - "auto": values like 8.5 display as 8.50%, values like 0.085 display as 8.50%
    - "decimal": values like 0.085 display as 8.50%
    - "whole_number": values like 8.5 display as 8.50%
    """
    formatted = df.copy()

    def _to_set(cols):
        if cols is None:
            return set()
        if isinstance(cols, str):
            return {cols}
        return set(cols)

    def _clean_name(col):
        return str(col).lower().replace("%", " pct ")

    def _has_any(col, keywords):
        name = _clean_name(col)
        return any(keyword in name for keyword in keywords)

    def _is_date_like_col(col):
        name = _clean_name(col).replace("_", " ")
        tokens = set(name.split())
        return (
            "date" in tokens
            or name.endswith("date")
            or "asof" in name
            or "as of" in name
            or "cutoff" in name
        )

    def _is_identifier(col):
        name = _clean_name(col).replace(" ", "_")
        return (
            name == "id"
            or name.endswith("_id")
            or "loan_id" in name
            or "account_id" in name
            or "contract_id" in name
            or "borrower_id" in name
        )

    def _infer_percent_multiplier(series):
        if percent_scale == "decimal":
            return 1
        if percent_scale == "whole_number":
            return 0.01
        if percent_scale != "auto":
            raise ValueError("percent_scale must be 'auto', 'decimal', or 'whole_number'")

        values = pd.to_numeric(series, errors="coerce").dropna().abs()
        if values.empty:
            return 1
        return 0.01 if values.quantile(0.90) > 1 and values.quantile(0.90) <= 100 else 1

    exclude = _to_set(exclude_cols)
    numeric_cols = [
        col
        for col in formatted.columns
        if pd.api.types.is_numeric_dtype(formatted[col]) and col not in exclude and not _is_identifier(col)
    ]

    percent_keywords = [
        "pct",
        "percent",
        "percentage",
        "rate",
        "yield",
        "coupon",
        "margin",
        "ltv",
        "dti",
        "apr",
        "cpr",
        "cdr",
        "dq",
        "delinquency",
        "default",
    ]
    accounting_keywords = [
        "net",
        "profit",
        "loss",
        "pnl",
        "variance",
        "difference",
        "delta",
        "change",
        "income",
        "expense",
    ]
    currency_keywords = [
        "amount",
        "balance",
        "bal",
        "principal",
        "interest",
        "payment",
        "recovery",
        "chargeoff",
        "charge_off",
        "price",
        "value",
        "revenue",
        "cost",
        "fee",
        "proceeds",
        "advance",
        "funded",
        "financed",
        "origination fee",
        "money deployed",
        "deployed",
        "upb",
    ]
    number_keywords = [
        "count",
        "num",
        "number",
        "term",
        "mob",
        "age",
        "fico",
        "score",
        "months",
    ]
    forced_percent = _to_set(percent_cols)
    forced_accounting = _to_set(accounting_cols)
    forced_currency = _to_set(currency_cols)
    forced_number = _to_set(number_cols)
    forced_date = _to_set(date_cols)
    forced_cols = forced_percent | forced_accounting | forced_currency | forced_number | forced_date

    date_set = forced_date | {
        col
        for col in formatted.columns
        if (
            col not in exclude
            and (
                pd.api.types.is_datetime64_any_dtype(formatted[col])
                or _is_date_like_col(col)
            )
        )
    }

    percent_set = forced_percent | {
        col for col in numeric_cols if col not in forced_cols and _has_any(col, percent_keywords)
    }
    accounting_set = forced_accounting | {
        col
        for col in numeric_cols
        if col not in forced_cols and col not in percent_set and _has_any(col, accounting_keywords)
    }
    currency_set = forced_currency | {
        col
        for col in numeric_cols
        if (
            col not in forced_cols
            and col not in percent_set
            and col not in accounting_set
            and _has_any(col, currency_keywords)
        )
    }
    number_set = forced_number | {
        col
        for col in numeric_cols
        if (
            col not in forced_cols
            and col not in percent_set
            and col not in accounting_set
            and col not in currency_set
            and _has_any(col, number_keywords)
        )
    }
    accounting_set -= percent_set
    currency_set -= percent_set | accounting_set
    number_set -= percent_set | accounting_set | currency_set
    numeric_display_cols = percent_set | accounting_set | currency_set | number_set

    for col in percent_set:
        if col in formatted.columns:
            multiplier = _infer_percent_multiplier(formatted[col])
            formatted[col] = formatted[col].apply(
                lambda x, m=multiplier: "" if pd.isna(x) else fmt_percent(x * m, percent_decimals)
            )

    for col in accounting_set:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: fmt_accounting(x, accounting_decimals, ""))

    for col in currency_set:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: fmt_currency(x, currency_decimals, ""))

    for col in number_set:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: fmt_number(x, number_decimals))

    for col in date_set:
        if col in formatted.columns:
            formatted[col] = pd.to_datetime(formatted[col], errors="coerce").dt.strftime(date_format)
            formatted[col] = formatted[col].fillna("")

    if return_styler:
        styler = formatted.style
        right_align_cols = [col for col in formatted.columns if col in numeric_display_cols]
        if right_align_cols:
            styler = styler.set_properties(
                subset=right_align_cols,
                **{"text-align": "right"},
            )
        if display_table:
            display(styler)
        return styler

    if display_table:
        display(formatted)

    return formatted


def map_naics_to_sector(naics):
    try:
        naics_str = str(naics).strip()  # clean numeric
        sector = int(naics_str[:2])
    except (TypeError, ValueError):
        return "Unknown"

    mapping = {
        (11,): "Agriculture",
        (21,): "Mining",
        (22,): "Utilities",
        (23,): "Construction",
        (31, 32, 33): "Manufacturing",
        (42,): "Wholesale Trade",
        (44, 45): "Retail Trade",
        (48, 49): "Transportation & Warehousing",
        (51,): "Information",
        (52,): "Finance & Insurance",
        (53,): "Real Estate and Rental and Leasing",
        (54,): "Professional, Scientific, and Technical Services",
        (55,): "Management of Companies and Enterprises",
        (56,): "Administrative and Support and Waste Management and Remediation Services",
        (61,): "Educational Services",
        (62,): "Healthcare",
        (71,): "Arts & Entertainment",
        (72,): "Accommodation & Food",
        (81,): "Other Services",
        (92,): "Public Administration"
    }

    for keys, value in mapping.items():
        if sector in keys:
            return value

    return "Other"


def get_wt_distribution(df, col, vintage, weight_by = 'Amount Financed', show_total_origination = False, format_pct = True, sort_by_col = None):

    wt_distribution_ = df.groupby([col, vintage])[[weight_by]].sum().unstack(fill_value=0)
    wt_distribution = wt_distribution_ / wt_distribution_.sum(axis=0)
    wt_distribution.columns = wt_distribution.columns.droplevel(0)
    if sort_by_col and sort_by_col in wt_distribution.columns:
        wt_distribution = wt_distribution.sort_values(by=sort_by_col, ascending=False)
    if format_pct:
        wt_distribution = wt_distribution.map(lambda x: f"{x:.2%}")
    if show_total_origination:
        wt_distribution.loc['Total Origination'] = np.array(wt_distribution_.sum(axis=0))
    
    return wt_distribution


def group_tail_categories(df, col, weight_col, max_categories=None, other_label="Other"):
    """
    Keep the largest categories by total weight and group the remaining categories.
    """
    if max_categories is None:
        return df

    if max_categories < 1:
        raise ValueError("max_categories must be at least 1.")

    out = df.copy()
    total_weight_by_category = out.groupby(col, dropna=False)[weight_col].sum()

    if len(total_weight_by_category) <= max_categories:
        return out

    top_categories = set(
        total_weight_by_category
        .sort_values(ascending=False)
        .head(max_categories)
        .index
    )

    out[col] = out[col].where(out[col].isin(top_categories), other_label)
    return out

def plot_distribution_vs_total_volume(
    distribution_df,
    total_volume_row="Total Origination",
    title="Distribution vs Total Origination Volume",
    x_label="",
    distribution_ylabel="Distribution",
    volume_ylabel="Total Origination Volume",
    category_label_format="{category}",
    figsize=(12, 6),
    volume_color="#d9e2ec",
    volume_edge_color="#8aa2b8",
    volume_alpha=0.45,
    marker="o",
    linewidth=2,
    legend_location="upper left",
    legend_bbox_to_anchor=None,
    legend_ncol=1,
    distribution_y_min=None,
    distribution_y_max=None,
    volume_y_min=None,
    volume_y_max=None,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    show=True,
):
    """
    Plot a distribution table with category weights on the left y-axis and
    total volume on the right y-axis.

    Expected input shape matches get_wt_distribution(..., show_total_origination=True):
    - index: categories plus a total volume row
    - columns: periods/vintages/months
    - category rows: percentages as decimals or strings like "12.3%"
    - total row: volume amounts
    """
    if total_volume_row not in distribution_df.index:
        raise ValueError(
            f"Expected '{total_volume_row}' row in distribution_df. "
            "Rebuild the table with show_total_origination=True or pass total_volume_row."
        )

    chart_df = distribution_df.copy()

    def _parse_number_series(series):
        return pd.to_numeric(
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False),
            errors="coerce",
        )

    def _parse_distribution_series(series):
        text_values = series.astype(str)
        numeric_values = _parse_number_series(series)
        non_null_values = numeric_values.dropna().abs()

        has_percent_sign = text_values.str.contains("%", regex=False).any()
        looks_like_whole_number_pct = (
            not non_null_values.empty
            and non_null_values.quantile(0.90) > 1
            and non_null_values.quantile(0.90) <= 100
        )
        if has_percent_sign or looks_like_whole_number_pct:
            numeric_values = numeric_values / 100

        return numeric_values

    total_volume = _parse_number_series(chart_df.loc[total_volume_row])
    distribution = chart_df.drop(index=total_volume_row).T
    distribution = distribution.apply(_parse_distribution_series)

    x_labels = total_volume.index.astype(str)
    x = np.arange(len(x_labels))

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    for category in distribution.columns:
        ax_left.plot(
            x,
            distribution[category].values,
            marker=marker,
            linewidth=linewidth,
            label=category_label_format.format(category=category),
            zorder=3,
        )

    ax_right.bar(
        x,
        total_volume.values,
        color=volume_color,
        edgecolor=volume_edge_color,
        alpha=volume_alpha,
        label=volume_ylabel,
        zorder=1,
    )

    ax_left.set_zorder(ax_right.get_zorder() + 1)
    ax_left.patch.set_visible(False)

    ax_left.set_title(title, fontsize=title_fontsize, weight="bold", loc="left")
    ax_left.set_xlabel(x_label, fontsize=label_fontsize)
    ax_left.set_ylabel(distribution_ylabel, fontsize=label_fontsize)
    ax_right.set_ylabel(volume_ylabel, fontsize=label_fontsize)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax_left.tick_params(axis="y", labelsize=tick_fontsize)
    ax_right.tick_params(axis="y", labelsize=tick_fontsize)
    ax_left.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_right.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax_left.grid(axis="y", alpha=0.25)

    if distribution_y_min is not None or distribution_y_max is not None:
        current_min, current_max = ax_left.get_ylim()
        ax_left.set_ylim(
            distribution_y_min if distribution_y_min is not None else current_min,
            distribution_y_max if distribution_y_max is not None else current_max,
        )

    if volume_y_min is not None or volume_y_max is not None:
        current_min, current_max = ax_right.get_ylim()
        ax_right.set_ylim(
            volume_y_min if volume_y_min is not None else current_min,
            volume_y_max if volume_y_max is not None else current_max,
        )

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        frameon=False,
        loc=legend_location,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
    )

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax_left, ax_right


def plot_distribution_stacked_bar(
    distribution_df,
    total_volume_row="Total Origination",
    title="Distribution",
    x_label="",
    distribution_ylabel="Distribution",
    category_label_format="{category}",
    figsize=(12, 6),
    colors=None,
    edge_color="white",
    edge_linewidth=0.4,
    legend_location="upper left",
    legend_bbox_to_anchor=None,
    legend_ncol=1,
    distribution_y_min=None,
    distribution_y_max=None,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=11,
    show=True,
):
    """
    Plot a weighted distribution table as stacked bars.

    If a total volume row is present, it is excluded from the stacked bars.
    """
    chart_df = distribution_df.copy()
    if total_volume_row in chart_df.index:
        chart_df = chart_df.drop(index=total_volume_row)

    def _parse_number_series(series):
        return pd.to_numeric(
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False),
            errors="coerce",
        )

    def _parse_distribution_series(series):
        text_values = series.astype(str)
        numeric_values = _parse_number_series(series)
        non_null_values = numeric_values.dropna().abs()

        has_percent_sign = text_values.str.contains("%", regex=False).any()
        looks_like_whole_number_pct = (
            not non_null_values.empty
            and non_null_values.quantile(0.90) > 1
            and non_null_values.quantile(0.90) <= 100
        )
        if has_percent_sign or looks_like_whole_number_pct:
            numeric_values = numeric_values / 100

        return numeric_values

    distribution = chart_df.T.apply(_parse_distribution_series).fillna(0)
    x_labels = distribution.index.astype(str)
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(distribution))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    if colors is None:
        colors = default_colors
    elif isinstance(colors, str):
        colors = [colors]
    else:
        colors = list(colors)
    if not colors:
        colors = default_colors

    for i, category in enumerate(distribution.columns):
        values = distribution[category].values
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors[i % len(colors)],
            edgecolor=edge_color,
            linewidth=edge_linewidth,
            label=category_label_format.format(category=category),
        )
        bottom = bottom + values

    ax.set_title(title, fontsize=title_fontsize, weight="bold", loc="left")
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(distribution_ylabel, fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.25)

    if distribution_y_min is not None or distribution_y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            distribution_y_min if distribution_y_min is not None else current_min,
            distribution_y_max if distribution_y_max is not None else current_max,
        )

    ax.legend(
        frameon=False,
        loc=legend_location,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
    )

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax


# def weighted_avg(df, cols, weight_col):
#     w = df[weight_col]
#     return df[cols].apply(lambda x: np.average(x, weights=w))

def weighted_avg(series, weights, exclude_nonpositive_values=True):
    mask = series.notna() & weights.notna() & (weights > 0)
    if exclude_nonpositive_values:
        mask = mask & (series > 0)
    if not mask.any():
        return np.nan
    return np.average(series[mask], weights=weights[mask])

def groupby_weighted_avg(df, groupby_col, cols, weight_col, exclude_nonpositive_values=True):

    result = (
    df.groupby(groupby_col)
      .apply(lambda g: pd.Series({
          col: weighted_avg(
              g[col],
              g[weight_col],
              exclude_nonpositive_values=exclude_nonpositive_values,
          )
          for col in cols
      }))
    )
    return result


def weighted_missing_pct(g, cols, weight_col, mode="zero_or_nan"):
    """
    Compute weighted % of:
        - zeros
        - NaNs
        - or both combined (default)

    Parameters:
    - g: grouped DataFrame
    - cols: list of columns to evaluate
    - weight_col: column used for weighting
    - mode: "zero_or_nan" (default), "zero", or "nan"
    """

    w = g[weight_col]

    # valid weights only
    valid_w = w.notna() & (w > 0)
    if not valid_w.any():
        return pd.Series({f"{col}_{mode}_pct": np.nan for col in cols})

    w = w[valid_w]
    total = w.sum()

    results = {}

    for col in cols:
        x = g.loc[valid_w, col]

        if mode == "zero":
            mask = (x == 0) & x.notna()
        elif mode == "nan":
            mask = x.isna()
        elif mode == "zero_or_nan":
            mask = (x == 0) | x.isna()
        else:
            raise ValueError("mode must be 'zero_or_nan', 'zero', or 'nan'")

        results[f"{col}_{mode}_pct"] = w[mask].sum() / total

    return pd.Series(results)


def groupby_weighted_missing(df, groupby_col, cols, weight_col, mode="zero_or_nan"):
    """
    Apply weighted_missing_pct across groups
    """
    return df.groupby(groupby_col).apply(
        lambda g: weighted_missing_pct(g, cols, weight_col, mode=mode)
    )


def adjust_lightness(color, factor): 
    c = mcolors.to_rgb(color)
    return tuple(min(1, max(0, x * factor)) for x in c)


def plot_finance_style(df, title="Chart Title", ylabel="", percentage=True, decimal = 2, data_labels=True,
                       xlabel = "", 
                       y_horizontal = None,
                       y_horizontal_label = '',
                       y_horizontal_position = 'right',
                       secondary_y_cols = None,              
                       secondary_ylabel = "",              
                       secondary_percentage = False,        
                       use_custom_colors = True,
                       color_groups_inputs = None,
                       linestyles_inputs = None,
                       figsize=(10, 5),

                        y_min=None,
                        y_max=None,
                        secondary_y_min=None,
                        secondary_y_max=None,
                        first_n_styles = None,

                        legend_cutoff_num = None,
                        legend_location = "upper left",
                        moving_average_window=None,
                        moving_average_min_periods=1,
                        trim_last_n=0

                       ):
    plt.style.use("default")

    df = df.copy()

    if moving_average_window is not None:
        if moving_average_window < 1:
            raise ValueError("moving_average_window must be at least 1.")
        if moving_average_min_periods < 1:
            raise ValueError("moving_average_min_periods must be at least 1.")
        df = df.rolling(
            window=moving_average_window,
            min_periods=moving_average_min_periods,
        ).mean()

    if trim_last_n is not None and trim_last_n > 0:
        df.iloc[-trim_last_n:, :] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    ### NEW: create second axis
    ax2 = ax.twinx() if secondary_y_cols else None

    if use_custom_colors == False:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        if color_groups_inputs:
            color_groups = color_groups_inputs
        else:
            color_groups = [
                ["#1f4e79", "#2b5d8a", "#3a6f9e", "#4f84b5"],
                ["#2f5d3a", "#3f7a4d", "#4f9961", "#6bb27a"],
                ["#7a1f1f", "#993333", "#b34d4d", "#cc6666"],
                ["#4b3b6d", "#6a4f9c", "#8a6cc2", "#b19cd9"],
                # ["#F106DD", "#595959", "#737373", "#8c8c8c"]
                # ["#404040", "#595959", "#737373", "#8c8c8c"]
            ]

        colors = []
        for i, col in enumerate(df.columns):
            group = i // 4
            shade = i % 4
            colors.append(color_groups[group % len(color_groups)][shade])

    for i, col in enumerate(df.columns):
        target_ax = ax2 if (secondary_y_cols and col in secondary_y_cols) else ax

        # ✅ Priority 1: custom styles for first n curves
        if first_n_styles and i < len(first_n_styles):
            style = first_n_styles[i]
            color = style.get("color", colors[i])
            linestyle = style.get("linestyle", "-")  # default solid
        else:
            color = colors[i]
            linestyle = (
                linestyles_inputs[i % len(linestyles_inputs)]
                if linestyles_inputs else "-"
            )

        if legend_cutoff_num!=None and i >= legend_cutoff_num:
            linestyle = ':'

        target_ax.plot(
            df.index.to_numpy(),
            df[col].to_numpy(),
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=col
        )

    ### NEW (Fix #1): force autoscale separately
    ax.relim()
    ax.autoscale_view()

    if ax2:
        ax2.relim()
        ax2.autoscale_view()

    # Title
    ax.set_title(title, fontsize=13, weight="bold", loc="left")

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ### NEW: secondary axis label
    if ax2:
        ax2.set_ylabel(secondary_ylabel)

    # Grid
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    ax.grid(axis="x", visible=False)

    # Remove clutter
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Primary axis formatter
    if percentage:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f"{{x:,.{decimal}%}}"))
    else:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(f"{{x:,.{decimal}f}}"))

    ### NEW (Fix #2): separate formatter for secondary axis
    if ax2:
        if secondary_percentage:
            ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter(f"{{x:,.{decimal}%}}"))
        else:
            ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter(f"{{x:,.{decimal}f}}"))

    # Format dates
    if hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        plt.xticks(rotation=0)

    ### MODIFIED: combine legends
    lines, labels = ax.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    if legend_cutoff_num != None:
        ax.legend(lines[:legend_cutoff_num], labels[:legend_cutoff_num], frameon=False, loc=legend_location)
    else:
        ax.legend(lines, labels, frameon=False, loc=legend_location)

    # Data labels
    if data_labels:
        for col in df.columns:
            valid = df[col][np.isfinite(df[col])]
            if not valid.empty:
                x = valid.index[-1]
                y = valid.iloc[-1]

                target_ax = ax2 if (secondary_y_cols and col in secondary_y_cols) else ax  ### NEW

                if percentage:
                    target_ax.text(x, y, f"{y:,.{decimal}%}", fontsize=9)
                else:
                    target_ax.text(x, y, f"{y:,.{decimal}f}", fontsize=9)

    # Horizontal lines (primary axis only)
    if y_horizontal is not None:
        x_min, x_max = ax.get_xlim()
        if y_horizontal_position == 'right':
            ha_ = 'right'
            x_pos = x_max - 0.5
        elif y_horizontal_position == 'left':
            ha_ = 'left'
            x_pos = x_min + 0.5
        else:
            print("Invalid y_horizontal_position. Use 'left' or 'right'. Defaulting to 'right'.")
            ha_ = 'right'
            x_pos = x_max - 0.5

        if isinstance(y_horizontal, (int, float)):
            ax.axhline(y=y_horizontal, color='red', linestyle='--', linewidth=2)
            ax.text(x=x_pos, y=y_horizontal,
                    s=y_horizontal_label + f'{y_horizontal:.2%}',
                    ha=ha_, va='bottom', color='red', fontsize=10)
        else:
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            for i in range(len(y_horizontal)):
                ax.axhline(y=y_horizontal[i], color=colors[i], linestyle='--', linewidth=2)
                ax.text(x=x_pos, y=y_horizontal[i],
                        s=y_horizontal_label[i] + f'{y_horizontal[i]:.2%}',
                        ha=ha_, va='bottom', color=colors[i], fontsize=10)

    ## explicitly decouple axis limits
    if ax2 and secondary_y_cols and df[secondary_y_cols].min().min() == 0:
        ax2.set_ylim(
            df[secondary_y_cols].min().min(),
            # df[secondary_y_cols].max().max()
        )

     # ✅ Apply custom y-axis limits (primary)
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )

    # ✅ Apply custom y-axis limits (secondary)
    if ax2 and (secondary_y_min is not None or secondary_y_max is not None):
        current_min, current_max = ax2.get_ylim()
        ax2.set_ylim(
            secondary_y_min if secondary_y_min is not None else current_min,
            secondary_y_max if secondary_y_max is not None else current_max
        )

    plt.tight_layout()
    plt.show()

def get_analytic_metric(df, metric, trim_n=3, rolling_window=None):
    df_metric = df[[metric]].unstack(level = 0)
    df_metric.columns = df_metric.columns.get_level_values(1)
    df_metric.replace(0, np.nan, inplace=True)
    df_metric = df_metric.apply(trim_last_n, n=trim_n)
    if rolling_window is not None:
        ''' Please note NANs are filled by zeroes before rolling sum'''
        # df_metric = df_metric.rolling(window=rolling_window).mean()

        df_metric = (
        df_metric
        .fillna(0)
        .rolling(rolling_window, min_periods=1)
        .sum()
        .pipe(lambda x: x.where(x != 0)
        / rolling_window
        ))
        df_metric = df_metric.apply(trim_last_n, n=rolling_window)
    return df_metric


def plot_distribution_with_density(df, col, bins=30):
    data = df[col].dropna()

    plt.figure()

    # Histogram
    plt.hist(data, bins=bins, density=True)

    # Density curve
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200)
    plt.plot(x_vals, kde(x_vals))

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


def calc_everd30_co(df, 
                    age_cutoff, 
                    dpd_col_name, 
                    dpd_num, 
                    mob_col_name, 
                    loan_id_col_name,
                    default_col_name):
    """
    df must have: loan_id, age_month, dpd, charged_off (bool or 0/1)
    age_cutoff: int (e.g., 12 for 12 months)
    """

    # Filter to observations BEFORE cutoff
    df_cut = df[df[mob_col_name] <= age_cutoff].copy()

    # Ever 30+ DPD before cutoff
    ever_d30 = (
        df_cut.assign(d30_flag=lambda x: x[dpd_col_name] >= dpd_num)
              .groupby(loan_id_col_name)["d30_flag"]
              .max()
              .rename("everD30")
    )

    # Charge-off before cutoff
    co_flag = (
        df_cut.assign(co_flag=lambda x: x[default_col_name] > 0)
              .groupby(loan_id_col_name)["co_flag"]
              .max()
              .rename("EverCO")
    )

    # Combine
    result = pd.concat([ever_d30, co_flag,  np.logical_or(ever_d30, co_flag)], axis=1).fillna(0)
    result.reset_index()
    result.columns = ['everD30', 'everCO', 'everD30 or CO']

    return result


def create_everDQ_MOBn_chart(df, 
                             age_cutoff = 3, 
                             dpd_col_name = "DaysDelinquent", 
                             default_col_name = 'Default',
                             dpd_num = 30, 
                             mob_col_name = 'MOB', 
                             loan_id_col_name = "LoanID",
                             original_bal_col_name = 'AmountFinanced',
                             vintage_col_name = 'FundingYearMonth',
                             mob_orig_data = 1,
                             filter_to_orig_data = False,
                             filter_mask = None
                             ):
    is_multiple_cutoffs = isinstance(age_cutoff, (list, tuple, set, np.ndarray, pd.Index, pd.Series))
    age_cutoffs = list(age_cutoff) if is_multiple_cutoffs else [age_cutoff]
    if not age_cutoffs:
        raise ValueError("age_cutoff must be a number or a non-empty list of numbers.")

    if filter_to_orig_data:
        if filter_mask is None:
            raise ValueError("filter_mask must be provided when filter_to_orig_data is True.")
        else:
            ever_dq_co = df[filter_mask].copy()
    else:
        ever_dq_co = df[df[mob_col_name] == mob_orig_data].copy()
    
    ever_dq_cols = []

    for cutoff in age_cutoffs:
        ever_dq_res = calc_everd30_co(
            df,
            cutoff,
            dpd_col_name,
            dpd_num,
            mob_col_name,
            loan_id_col_name,
            default_col_name,
        )
        if loan_id_col_name not in ever_dq_res.columns:
            ever_dq_res = ever_dq_res.reset_index()

        ever_dq_col = 'EverD{:.0f} @ MOB{:.0f}'.format(dpd_num, cutoff)
        ever_dq_cols.append(ever_dq_col)

        if is_multiple_cutoffs:
            suffix = ' @ MOB{:.0f}'.format(cutoff)
            ever_flag_col = 'everD{:.0f} or CO{}'.format(dpd_num, suffix)
            ever_dq_res = ever_dq_res.rename(
                columns={
                    'everD30': 'everD{:.0f}{}'.format(dpd_num, suffix),
                    'everCO': 'everCO{}'.format(suffix),
                    'everD30 or CO': ever_flag_col,
                }
            )
        else:
            ever_flag_col = 'everD30 or CO'

        ever_dq_co = ever_dq_co.merge(ever_dq_res, on=loan_id_col_name, how='left')
        ever_flag = pd.to_numeric(ever_dq_co[ever_flag_col], errors='coerce').fillna(0)
        original_balance = to_numeric_amount(ever_dq_co[original_bal_col_name]).fillna(0)
        ever_dq_co[original_bal_col_name] = original_balance
        ever_dq_co[ever_dq_col] = ever_flag * original_balance

    out = ever_dq_co[[original_bal_col_name, vintage_col_name] + ever_dq_cols].groupby(vintage_col_name).sum(numeric_only=True)
    for cutoff, ever_dq_col in zip(age_cutoffs, ever_dq_cols):
        ever_dq_pct_col = 'EverD{:.0f}% @ MOB{:.0f}'.format(dpd_num, cutoff)
        out[ever_dq_pct_col] = out[ever_dq_col] / out[original_bal_col_name]

    return out, ever_dq_co

def read_in_files(DATA_PATH, sheet_name=0, skiprows=0, display_df=True, **read_kwargs):
    """
    Read a tabular file into a DataFrame for notebook/Data Wrangler use.

    Supported file types come from read_tabular:
    - Excel: .xlsx, .xlsm, .xls
    - CSV/text: .csv, .txt
    - Columnar: .parquet, .feather

    sheet_name is used for Excel files only.
    skiprows is used for Excel, CSV, and text files only.
    Extra keyword arguments are passed through to the underlying pandas reader.
    """
    DATA_PATH = Path(DATA_PATH)
    suffix = DATA_PATH.suffix.lower()

    reader_kwargs = dict(read_kwargs)

    if suffix in [".xlsx", ".xlsm", ".xls"]:
        reader_kwargs.setdefault("sheet_name", sheet_name)
        reader_kwargs.setdefault("skiprows", skiprows)
    elif suffix in [".csv", ".txt"]:
        reader_kwargs.setdefault("skiprows", skiprows)
    elif suffix in [".parquet", ".feather"]:
        pass
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df_raw = read_tabular(DATA_PATH, **reader_kwargs)

    data_wrangler_df = df_raw.copy()

    if display_df:
        print(f'Shape of the dataframe: {data_wrangler_df.shape}')
        display(data_wrangler_df)

    return data_wrangler_df


def to_numeric_amount(series):
    """
    Convert messy amount-like values to numeric.

    Handles strings with $, commas, percent signs, blanks, and accounting
    parentheses. Non-numeric markers such as "x" become NaN.
    """
    text = series.astype("string").str.strip()
    negative_mask = text.str.match(r"^\(.*\)$", na=False)

    cleaned = (
        text.str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )

    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric.where(~negative_mask, -numeric)


def add_ratio_columns(
    df,
    numerator_cols,
    denominator_col,
    output_suffix=" %",
    return_invalid_rows=False,
):
    """
    Add ratio columns from one denominator and one or more numerator columns.

    numerator_cols can be:
    - list of column names: ["fee"]
    - dict mapping source to output name: {"fee": "Fee %"}
    """
    out = df.copy()

    if isinstance(numerator_cols, dict):
        numerator_map = numerator_cols
    elif isinstance(numerator_cols, str):
        numerator_map = {numerator_cols: f"{numerator_cols}{output_suffix}"}
    else:
        numerator_map = {col: f"{col}{output_suffix}" for col in numerator_cols}

    missing_cols = [
        col for col in list(numerator_map.keys()) + [denominator_col] if col not in out.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing columns for ratio calculation: {missing_cols}")

    denominator = to_numeric_amount(out[denominator_col]).replace(0, np.nan)
    invalid_mask = denominator.isna()

    for numerator_col, output_col in numerator_map.items():
        numerator = to_numeric_amount(out[numerator_col])
        out[output_col] = numerator / denominator
        invalid_mask = invalid_mask | numerator.isna()

    if not return_invalid_rows:
        return out

    review_cols = list(numerator_map.keys()) + [denominator_col]
    invalid_rows = out.loc[invalid_mask, review_cols]
    return out, invalid_rows


def filter_existing_columns(df, cols):
    """
    Return nonblank columns that exist in df, preserving the input order.
    """
    if isinstance(cols, str):
        cols = [cols]

    seen = set()
    existing_cols = []
    for col in cols:
        if not col or col in seen or col not in df.columns:
            continue
        existing_cols.append(col)
        seen.add(col)

    return existing_cols


def format_column_label(col):
    """
    Convert a column name into a chart-friendly label.

    Examples:
    - credit_tier -> Credit Tier
    - dti_ratio -> DTI Ratio
    - current_upb -> Current UPB
    - borrower_income_frequency_clean -> Borrower Income Frequency
    """
    acronym_map = {
        "apr": "APR",
        "dti": "DTI",
        "fico": "FICO",
        "id": "ID",
        "ltv": "LTV",
        "upb": "UPB",
        "wa": "WA",
    }

    cleaned = str(col).replace("_", " ").replace("-", " ").strip()
    words = [word for word in cleaned.split() if word]
    if words and words[-1].lower() == "clean":
        words = words[:-1]
    formatted_words = [
        acronym_map.get(word.lower(), word[:1].upper() + word[1:].lower())
        for word in words
    ]
    return " ".join(formatted_words)


def _coerce_weighted_avg_series(series):
    numeric = to_numeric_amount(series)
    text_values = series.astype("string")
    if text_values.str.contains("%", regex=False, na=False).any():
        numeric = numeric / 100
    return numeric


def _set_weighted_avg_axis_formatter(ax, values, col, value_y_formatter=None):
    if value_y_formatter == "percent":
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        return
    if value_y_formatter == "number":
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.2f}"))
        return
    if value_y_formatter in ["integer", "whole_number"]:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        return
    if value_y_formatter is not None:
        ax.yaxis.set_major_formatter(value_y_formatter)
        return

    name = str(col).lower().replace("_", " ")
    percent_keywords = ["%", "pct", "percent", "rate", "ratio", "ltv", "dti", "apr"]
    amount_keywords = [
        "amount",
        "balance",
        "bal",
        "income",
        "payment",
        "fee",
        "upb",
        "volume",
        "principal",
    ]
    numeric_values = pd.Series(values).dropna().abs()

    if any(keyword in name for keyword in percent_keywords):
        if numeric_values.empty or numeric_values.quantile(0.90) <= 1.5:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        else:
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.2f}%"))
    elif any(keyword in name for keyword in amount_keywords):
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.2f}"))


def plot_metric_vs_volume(
    df,
    value_cols,
    volume_col,
    x_col=None,
    right_value_cols=None,
    title=None,
    x_label=None,
    value_ylabel=None,
    right_ylabel=None,
    volume_ylabel=None,
    figsize=(12, 6),
    volume_color="#d9e2ec",
    volume_edge_color="#8aa2b8",
    volume_alpha=0.45,
    marker="o",
    linewidth=2,
    right_linestyle="--",
    right_colors=None,
    legend_location="upper left",
    value_y_min=None,
    value_y_max=None,
    right_y_min=None,
    right_y_max=None,
    volume_y_min=None,
    volume_y_max=None,
    value_y_formatter=None,
    right_y_formatter=None,
    volume_display="background",
    format_value_labels=True,
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=12,
    legend_fontsize=12,
    legend=True,
    show=True,
):
    """
    Plot one or more metric columns on the left axis with optional right-axis
    metrics and volume/balance bars.

    volume_col can be one column name or a list of column names. When multiple
    volume columns are passed, they are plotted as grouped bars.

    volume_color and volume_edge_color can be a single color, a list of colors,
    or a dict keyed by volume column name.

    volume_display:
    - "background": volume bars sit behind the lines with the volume axis hidden.
      This is the default because origination volume is usually context, not
      the main metric.
    - "right_axis": volume bars use a visible right axis. If right_value_cols
      are provided, volume uses an extra right-side axis.
    - "none": volume is not plotted.

    Use this directly for already-built tables, such as everD30 rollups:
    plot_metric_vs_volume(
        everd30_table,
        value_cols=[everd30_pct_col],
        volume_col="original_loan_balance",
        right_value_cols="WA fico_score",
        volume_display="background",
        value_y_formatter="percent",
    )
    """
    def _as_list(cols):
        if cols is None:
            return []
        if isinstance(cols, str):
            return [cols]
        return list(cols)

    value_cols = _as_list(value_cols)
    right_value_cols = _as_list(right_value_cols)
    volume_cols = _as_list(volume_col)
    if not value_cols:
        raise ValueError("value_cols must include at least one column.")

    volume_display_aliases = {
        "right": "right_axis",
        "right_axis": "right_axis",
        "right_y_axis": "right_axis",
        "background": "background",
        "back": "background",
        "hidden": "background",
        "none": "none",
        None: "none",
    }
    if volume_display not in volume_display_aliases:
        raise ValueError("volume_display must be 'right_axis', 'background', or 'none'.")
    volume_display = volume_display_aliases[volume_display]
    if volume_display != "none" and not volume_cols:
        raise ValueError("volume_col must include at least one column unless volume_display is 'none'.")

    required_cols = list(value_cols) + list(right_value_cols)
    if volume_display != "none":
        required_cols.extend(volume_cols)
    if x_col is not None:
        required_cols.append(x_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for plotting: {missing_cols}")

    def _display_label(col, format_label=True):
        return format_column_label(col) if format_label else str(col)

    def _pick_color(colors, index, default_colors, key=None):
        if colors is None:
            return default_colors[index % len(default_colors)]
        if isinstance(colors, dict):
            return colors.get(key, default_colors[index % len(default_colors)])
        if isinstance(colors, str):
            return colors
        colors = list(colors)
        if not colors:
            return default_colors[index % len(default_colors)]
        return colors[index % len(colors)]

    def _as_plot_numeric(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")

    plot_df = df.copy()
    if volume_display != "none":
        for col in volume_cols:
            plot_df[col] = _as_plot_numeric(to_numeric_amount(plot_df[col]))
    for col in value_cols + right_value_cols:
        plot_df[col] = _as_plot_numeric(_coerce_weighted_avg_series(plot_df[col]))

    x_values = plot_df[x_col] if x_col is not None else plot_df.index
    x_labels = x_values.astype(str)
    x = np.arange(len(x_labels))
    if len(volume_cols) == 1:
        volume_label = volume_ylabel or format_column_label(volume_cols[0])
    else:
        volume_label = volume_ylabel or "Volume"
    x_axis_label = format_column_label(x_col) if x_col and x_label is None else (x_label or "")

    column_labels = [_display_label(col, format_value_labels) for col in value_cols]
    columns_label = ", ".join(column_labels)
    right_column_labels = [format_column_label(col) for col in right_value_cols]
    right_columns_label = ", ".join(right_column_labels)
    first_col = value_cols[0]
    first_col_label = column_labels[0]
    chart_title = (
        f"{columns_label} vs {right_columns_label or volume_label}"
        if title is None
        else title.format(
            column=first_col_label,
            column_name=first_col,
            columns=columns_label,
            right_columns=right_columns_label,
            volume=volume_label,
        )
    )
    if value_ylabel is None:
        chart_value_ylabel = first_col_label if len(value_cols) == 1 else "Value"
    else:
        chart_value_ylabel = value_ylabel.format(
            column=first_col_label,
            column_name=first_col,
            columns=columns_label,
        )
    if right_value_cols:
        first_right_col = right_value_cols[0]
        first_right_label = right_column_labels[0]
        if right_ylabel is None:
            chart_right_ylabel = first_right_label if len(right_value_cols) == 1 else "Right Axis"
        else:
            chart_right_ylabel = right_ylabel.format(
                column=first_right_label,
                column_name=first_right_col,
                columns=right_columns_label,
            )

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = None
    ax_volume = None

    if volume_display != "none":
        ax_volume = ax_left.twinx()
        volume_default_colors = ["#d9e2ec", "#b7c8d8", "#93a9bb", "#cbd5a1", "#dfb982"]
        volume_default_edge_colors = ["#8aa2b8", "#6f879b", "#566f82", "#8f985f", "#9e7653"]
        volume_color_spec = volume_color
        volume_edge_color_spec = volume_edge_color
        if len(volume_cols) > 1 and isinstance(volume_color, str) and volume_color == "#d9e2ec":
            volume_color_spec = None
        if len(volume_cols) > 1 and isinstance(volume_edge_color, str) and volume_edge_color == "#8aa2b8":
            volume_edge_color_spec = None
        bar_width = 0.8 / max(len(volume_cols), 1)
        bar_offsets = (np.arange(len(volume_cols)) - (len(volume_cols) - 1) / 2) * bar_width

        for i, col in enumerate(volume_cols):
            if len(volume_cols) == 1:
                bar_label = volume_label
            else:
                bar_label = format_column_label(col)

            ax_volume.bar(
                x + bar_offsets[i],
                plot_df[col].values,
                width=bar_width,
                color=_pick_color(volume_color_spec, i, volume_default_colors, key=col),
                edgecolor=_pick_color(volume_edge_color_spec, i, volume_default_edge_colors, key=col),
                alpha=volume_alpha,
                label=bar_label,
                zorder=1,
            )
        ax_volume.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

        if volume_display == "background":
            ax_volume.tick_params(axis="y", right=False, labelright=False)
            ax_volume.yaxis.set_visible(False)
            ax_volume.spines["right"].set_visible(False)
            ax_volume.set_ylabel("")
        elif right_value_cols:
            ax_volume.spines["right"].set_position(("axes", 1.10))
            ax_volume.set_ylabel(volume_label, fontsize=label_fontsize)
            ax_volume.tick_params(axis="y", labelsize=tick_fontsize)
        else:
            ax_right = ax_volume
            ax_right.set_ylabel(volume_label, fontsize=label_fontsize)
            ax_right.tick_params(axis="y", labelsize=tick_fontsize)

    for col in value_cols:
        col_label = _display_label(col, format_value_labels)
        ax_left.plot(
            x,
            plot_df[col].values,
            marker=marker,
            linewidth=linewidth,
            label=col_label,
            zorder=3,
        )

    if right_value_cols:
        ax_right = ax_left.twinx()
        right_default_colors = ["#7a5195", "#ef5675", "#ffa600", "#2f9e44"]
        right_axis_color = _pick_color(right_colors, 0, right_default_colors)
        for i, col in enumerate(right_value_cols):
            col_label = format_column_label(col)
            color = _pick_color(right_colors, i, right_default_colors)
            ax_right.plot(
                x,
                plot_df[col].values,
                color=color,
                marker=marker,
                linewidth=linewidth,
                linestyle=right_linestyle,
                label=col_label,
                zorder=4,
            )
        ax_right.set_ylabel(chart_right_ylabel, fontsize=label_fontsize)
        ax_right.yaxis.label.set_color(right_axis_color)
        ax_right.tick_params(axis="y", labelsize=tick_fontsize, colors=right_axis_color)
        ax_right.spines["right"].set_color(right_axis_color)
        _set_weighted_avg_axis_formatter(
            ax_right,
            plot_df[first_right_col],
            first_right_col,
            value_y_formatter=right_y_formatter,
        )

    if ax_volume is not None:
        ax_volume.set_zorder(0)
        ax_volume.patch.set_visible(False)
    if ax_right is not None:
        ax_right.set_zorder(3)
        ax_right.patch.set_visible(False)
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    ax_left.set_title(chart_title, fontsize=title_fontsize, weight="bold", loc="left")
    ax_left.set_xlabel(x_axis_label, fontsize=label_fontsize)
    ax_left.set_ylabel(chart_value_ylabel, fontsize=label_fontsize)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=tick_fontsize)
    ax_left.tick_params(axis="y", labelsize=tick_fontsize)
    _set_weighted_avg_axis_formatter(
        ax_left,
        plot_df[first_col],
        first_col,
        value_y_formatter=value_y_formatter,
    )
    ax_left.grid(axis="y", alpha=0.25)

    if value_y_min is not None or value_y_max is not None:
        current_min, current_max = ax_left.get_ylim()
        ax_left.set_ylim(
            value_y_min if value_y_min is not None else current_min,
            value_y_max if value_y_max is not None else current_max,
        )

    if right_value_cols and (right_y_min is not None or right_y_max is not None):
        current_min, current_max = ax_right.get_ylim()
        ax_right.set_ylim(
            right_y_min if right_y_min is not None else current_min,
            right_y_max if right_y_max is not None else current_max,
        )

    if ax_volume is not None and (volume_y_min is not None or volume_y_max is not None):
        current_min, current_max = ax_volume.get_ylim()
        ax_volume.set_ylim(
            volume_y_min if volume_y_min is not None else current_min,
            volume_y_max if volume_y_max is not None else current_max,
        )

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels() if ax_right is not None else ([], [])
    volume_handles, volume_labels = ax_volume.get_legend_handles_labels() if ax_volume is not None else ([], [])

    legend_handles = []
    legend_labels = []
    for handles, labels in [
        (left_handles, left_labels),
        (right_handles, right_labels),
        (volume_handles, volume_labels),
    ]:
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    if legend:
        ax_left.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc=legend_location,
            fontsize=legend_fontsize,
        )

    plt.tight_layout()
    if show:
        plt.show()

    chart = {
        "figure": fig,
        "value_axis": ax_left,
        "weighted_avg_axis": ax_left,
        "right_axis": ax_right,
        "right_value_axis": ax_right if right_value_cols else None,
        "right_metric_axis": ax_right if right_value_cols else None,
        "volume_axis": ax_volume,
        "volume_bar_axis": ax_volume,
    }
    charts = {"chart": chart}
    charts.update({col: chart for col in value_cols})
    charts.update({col: chart for col in right_value_cols})
    charts.update({col: chart for col in volume_cols})

    return charts

def plot_weighted_avg_by_group(
    df,
    groupby_col,
    cols_to_analyze,
    weight_col,
    total_volume_col="Total Origination Volume",
    title=None,
    x_label=None,
    value_ylabel=None,
    volume_ylabel="Total Origination Volume",
    figsize=(12, 6),
    volume_color="#d9e2ec",
    volume_edge_color="#8aa2b8",
    volume_alpha=0.45,
    marker="o",
    linewidth=2,
    right_colors=None,
    legend_location="upper left",
    value_y_min=None,
    value_y_max=None,
    volume_y_min=None,
    volume_y_max=None,
    value_y_formatter=None,
    volume_display="background",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=12,
    legend_fontsize=12,
    exclude_nonpositive_values=False,
    show=True,
    return_results=False,
):
    """
    Plot weighted-average numeric metrics by group against total volume.

    The function returns the weighted-average table by default. Pass
    return_results=True to also get the generated figure/axis objects.

    exclude_nonpositive_values=False keeps valid zero values, such as 0% fees,
    in the weighted average. Set it to True for fields where zero means missing.
    """
    missing_required = [col for col in [groupby_col, weight_col] if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if isinstance(cols_to_analyze, str):
        cols_to_analyze = [cols_to_analyze]

    cols_to_analyze = filter_existing_columns(df, cols_to_analyze)
    if not cols_to_analyze:
        raise ValueError("No cols_to_analyze were found in the DataFrame.")

    required_cols = filter_existing_columns(df, [groupby_col, weight_col] + cols_to_analyze)
    work = df[required_cols].copy()
    work[weight_col] = _coerce_weighted_avg_series(work[weight_col])

    usable_cols = []
    dropped_cols = []
    for col in cols_to_analyze:
        work[col] = _coerce_weighted_avg_series(work[col])
        if work[col].notna().any():
            usable_cols.append(col)
        else:
            dropped_cols.append(col)

    if not usable_cols:
        raise ValueError("None of cols_to_analyze could be converted to numeric values.")

    weighted_avg_table = groupby_weighted_avg(
        work,
        groupby_col,
        usable_cols,
        weight_col,
        exclude_nonpositive_values=exclude_nonpositive_values,
    )
    total_volume = work.groupby(groupby_col)[weight_col].sum(min_count=1).rename(total_volume_col)
    output_table = weighted_avg_table.join(total_volume, how="left")

    x_axis_label = format_column_label(groupby_col) if x_label is None else x_label
    charts = {}
    for col in usable_cols:
        col_charts = plot_metric_vs_volume(
            output_table,
            value_cols=col,
            volume_col=total_volume_col,
            title=title or "{column} Weighted Average vs {volume}",
            x_label=x_axis_label,
            value_ylabel=value_ylabel or "{column} Weighted Average",
            volume_ylabel=volume_ylabel,
            figsize=figsize,
            volume_color=volume_color,
            volume_edge_color=volume_edge_color,
            volume_alpha=volume_alpha,
            marker=marker,
            linewidth=linewidth,
            right_colors=right_colors,
            legend_location=legend_location,
            value_y_min=value_y_min,
            value_y_max=value_y_max,
            volume_y_min=volume_y_min,
            volume_y_max=volume_y_max,
            value_y_formatter=value_y_formatter,
            volume_display=volume_display,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            show=show,
        )
        charts[col] = col_charts[col]

    if return_results:
        return {
            "table": output_table,
            "charts": charts,
            "dropped_columns": dropped_cols,
        }

    return output_table


def plot_weighted_missing_by_group(
    df,
    groupby_col,
    cols_to_analyze,
    weight_col,
    mode="zero_or_nan",
    total_volume_col="Total Origination Volume",
    title=None,
    x_label=None,
    value_ylabel=None,
    volume_ylabel="Total Origination Volume",
    figsize=(12, 6),
    volume_color="#d9e2ec",
    volume_edge_color="#8aa2b8",
    volume_alpha=0.45,
    marker="o",
    linewidth=2,
    right_colors=None,
    legend_location="upper left",
    value_y_min=None,
    value_y_max=None,
    volume_y_min=None,
    volume_y_max=None,
    value_y_formatter="percent",
    volume_display="background",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=12,
    legend_fontsize=12,
    legend=True,
    show=True,
    return_results=False,
):
    """
    Plot weighted missing/zero rates by group against total volume.

    mode controls the definition of missing:
    - "zero_or_nan": weighted share where value is zero or null
    - "zero": weighted share where value is zero
    - "nan": weighted share where value is null
    """
    if mode not in ["zero_or_nan", "zero", "nan"]:
        raise ValueError("mode must be 'zero_or_nan', 'zero', or 'nan'")

    missing_required = [col for col in [groupby_col, weight_col] if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if isinstance(cols_to_analyze, str):
        cols_to_analyze = [cols_to_analyze]

    cols_to_analyze = filter_existing_columns(df, cols_to_analyze)
    if not cols_to_analyze:
        raise ValueError("No cols_to_analyze were found in the DataFrame.")

    required_cols = filter_existing_columns(df, [groupby_col, weight_col] + cols_to_analyze)
    work = df[required_cols].copy()
    work[weight_col] = to_numeric_amount(work[weight_col])

    weighted_missing_table = groupby_weighted_missing(
        work,
        groupby_col,
        cols_to_analyze,
        weight_col,
        mode=mode,
    )
    total_volume = work.groupby(groupby_col)[weight_col].sum(min_count=1).rename(total_volume_col)
    output_table = weighted_missing_table.join(total_volume, how="left")

    x_axis_label = format_column_label(groupby_col) if x_label is None else x_label
    charts = {}
    for col in cols_to_analyze:
        metric_col = f"{col}_{mode}_pct"
        col_label = format_column_label(col)
        col_charts = plot_metric_vs_volume(
            output_table,
            value_cols=metric_col,
            volume_col=total_volume_col,
            title=title or f"{col_label} Weighted Missing Rate vs {{volume}}",
            x_label=x_axis_label,
            value_ylabel=value_ylabel or f"{col_label} Weighted Missing Rate",
            volume_ylabel=volume_ylabel,
            figsize=figsize,
            volume_color=volume_color,
            volume_edge_color=volume_edge_color,
            volume_alpha=volume_alpha,
            marker=marker,
            linewidth=linewidth,
            right_colors=right_colors,
            legend_location=legend_location,
            value_y_min=value_y_min,
            value_y_max=value_y_max,
            volume_y_min=volume_y_min,
            volume_y_max=volume_y_max,
            value_y_formatter=value_y_formatter,
            volume_display=volume_display,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            legend=legend,
            show=show,
        )
        charts[col] = col_charts[metric_col]

    if return_results:
        return {
            "table": output_table,
            "charts": charts,
        }

    return output_table


def plot_distribution_for_categorical(
    df,
    cols_to_analyze,
    vintage_col,
    weight_col,
    total_volume_row="Total Origination",
    x_label=None,
    volume_ylabel="Total Origination Volume",
    category_label_format="{category}",
    figsize=(12, 6),
    plot_kind="line",
    sort_by_col=None,
    format_pct=True,
    show=True,
    max_categories=None,
    other_label="Other",
    **plot_kwargs,
):
    """
    Build weighted distribution tables for categorical columns and plot each one
    against total origination volume.

    The input to plot_distribution_vs_total_volume is created inside this
    function using get_wt_distribution(..., show_total_origination=True).

    If max_categories is provided, each analyzed column keeps its largest
    categories by total weight and groups the rest into other_label.

    plot_kind can be "line" for the existing line-plus-volume chart or
    "stacked_bar" for stacked distribution bars without the volume axis.
    """
    if isinstance(cols_to_analyze, str):
        cols_to_analyze = [cols_to_analyze]

    plot_kind_aliases = {
        "line": "line",
        "lines": "line",
        "volume": "line",
        "stacked": "stacked_bar",
        "stacked_bar": "stacked_bar",
        "bar": "stacked_bar",
    }
    if plot_kind not in plot_kind_aliases:
        raise ValueError("plot_kind must be 'line' or 'stacked_bar'.")
    plot_kind = plot_kind_aliases[plot_kind]

    results = {}
    x_axis_label = format_column_label(vintage_col) if x_label is None else x_label

    for col in cols_to_analyze:
        col_label = format_column_label(col)
        plot_df = group_tail_categories(
            df,
            col,
            weight_col,
            max_categories=max_categories,
            other_label=other_label,
        )

        distribution_table = get_wt_distribution(
            plot_df,
            col,
            vintage_col,
            weight_by=weight_col,
            show_total_origination=True,
            format_pct=format_pct,
            sort_by_col=sort_by_col,
        )

        if plot_kind == "stacked_bar":
            fig, ax_left = plot_distribution_stacked_bar(
                distribution_table,
                total_volume_row=total_volume_row,
                title=f"{col_label} Distribution",
                x_label=x_axis_label,
                distribution_ylabel=f"{col_label} Distribution",
                category_label_format=category_label_format,
                figsize=figsize,
                show=show,
                **plot_kwargs,
            )
            ax_right = None
        else:
            fig, ax_left, ax_right = plot_distribution_vs_total_volume(
                distribution_table,
                total_volume_row=total_volume_row,
                title=f"{col_label} Distribution vs Total Origination Volume",
                x_label=x_axis_label,
                distribution_ylabel=f"{col_label} Distribution",
                volume_ylabel=volume_ylabel,
                category_label_format=category_label_format,
                figsize=figsize,
                show=show,
                **plot_kwargs,
            )

        results[col] = {
            "table": distribution_table,
            "figure": fig,
            "distribution_axis": ax_left,
            "volume_axis": ax_right,
        }

    return results


def fill_from_latest_nonblank_snapshot(
    df,
    id_col,
    snapshot_col,
    fill_col,
    blank_regex=r"^\s*$",
    inplace=False,
    display_audit=True,
):
    """
    Fill blank values in fill_col using the latest non-blank value from the same ID.

    Also adds audit columns:
    - {fill_col}_was_filled
    - {fill_col}_fill_source_snapshot_date
    - {fill_col}_fill_source_not_latest_snapshot
    """
    out = df if inplace else df.copy()

    out[snapshot_col] = pd.to_datetime(out[snapshot_col])
    out[fill_col] = out[fill_col].replace(blank_regex, pd.NA, regex=True)

    latest_snapshot_by_id = (
        out.groupby(id_col)[snapshot_col]
        .max()
        .rename("latest_snapshot_date")
    )

    latest_nonblank = (
        out[out[fill_col].notna()]
        .sort_values([id_col, snapshot_col])
        .groupby(id_col)
        .tail(1)
        .set_index(id_col)[[snapshot_col, fill_col]]
        .rename(columns={
            snapshot_col: f"{fill_col}_source_snapshot_date",
            fill_col: f"{fill_col}_latest_nonblank_value",
        })
    )

    latest_nonblank = latest_nonblank.join(latest_snapshot_by_id)

    latest_nonblank[f"{fill_col}_source_not_latest_snapshot"] = (
        latest_nonblank[f"{fill_col}_source_snapshot_date"]
        < latest_nonblank["latest_snapshot_date"]
    )

    missing_mask = out[fill_col].isna()

    out.loc[missing_mask, fill_col] = (
        out.loc[missing_mask, id_col]
        .map(latest_nonblank[f"{fill_col}_latest_nonblank_value"])
    )

    out[f"{fill_col}_was_filled"] = missing_mask
    out[f"{fill_col}_fill_source_snapshot_date"] = out[id_col].map(
        latest_nonblank[f"{fill_col}_source_snapshot_date"]
    )
    out[f"{fill_col}_fill_source_not_latest_snapshot"] = out[id_col].map(
        latest_nonblank[f"{fill_col}_source_not_latest_snapshot"]
    )

    audit_table = latest_nonblank.reset_index()[[
        id_col,
        f"{fill_col}_latest_nonblank_value",
        f"{fill_col}_source_snapshot_date",
        "latest_snapshot_date",
        f"{fill_col}_source_not_latest_snapshot",
    ]]

    if display_audit:
        display(audit_table)

    return out, audit_table


def plot_correlation_simple(df, cols_for_corr, plot = True):
    cols_for_corr = [col for col in cols_for_corr if df[col].dtype != 'object' ]  

    corr = df[cols_for_corr].corr(method='pearson')
    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Matrix")
        plt.show()

    return corr


def get_everd_co_table(df,
                        everdqco_col,
                        everd_col_output_col_name,
                        groupby_col,
                        weight_col,
                        total_volume_col="Total Origination Volume",
                        category_col=None,
                        categories_to_plot=None,
                        max_categories=None,
                        other_label="Other",
                        category_missing_label="Missing"):
    required_cols = [groupby_col, weight_col, everdqco_col]
    if category_col is not None:
        required_cols.append(category_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work = df[required_cols].copy()
    work[weight_col] = to_numeric_amount(work[weight_col])
    work[everdqco_col] = pd.to_numeric(work[everdqco_col], errors="coerce").fillna(0)

    if category_col is not None:
        work[category_col] = work[category_col].astype("object").where(
            work[category_col].notna(),
            category_missing_label,
        )

        if categories_to_plot is not None:
            if isinstance(categories_to_plot, str):
                categories_to_plot = [categories_to_plot]
            categories_to_plot = list(categories_to_plot)
            keep_categories = set(categories_to_plot)
        elif max_categories is not None:
            if max_categories < 1:
                raise ValueError("max_categories must be at least 1.")
            categories_to_plot = (
                work.groupby(category_col, dropna=False)[weight_col]
                .sum()
                .sort_values(ascending=False)
                .head(max_categories)
                .index
                .tolist()
            )
            keep_categories = set(categories_to_plot)
        else:
            categories_to_plot = (
                work.groupby(category_col, dropna=False)[weight_col]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            keep_categories = set(categories_to_plot)

        work[category_col] = work[category_col].where(
            work[category_col].isin(keep_categories),
            other_label,
        )

    everd_weighted_col = f"{everdqco_col}_weighted"
    work[everd_weighted_col] = work[everdqco_col] * work[weight_col]

    if category_col is not None:
        groupeddf = (
            work.groupby([groupby_col, category_col], dropna=False)[[everd_weighted_col, weight_col]]
            .sum()
        )
        groupeddf["everd_rate"] = (
            groupeddf[everd_weighted_col] / groupeddf[weight_col].replace(0, np.nan)
        )

        rate_table = groupeddf["everd_rate"].unstack(category_col)
        category_order = [category for category in categories_to_plot if category in rate_table.columns]
        if other_label in rate_table.columns and other_label not in category_order:
            category_order.append(other_label)
        category_order.extend([category for category in rate_table.columns if category not in category_order])
        rate_table = rate_table[category_order]
        total_volume = work.groupby(groupby_col, dropna=False)[weight_col].sum().rename(total_volume_col)
        return rate_table.join(total_volume, how="left")

    groupeddf = work.groupby(groupby_col, dropna=False)[[everd_weighted_col, weight_col]].sum()
    groupeddf = groupeddf.rename(columns={weight_col: total_volume_col})
    groupeddf[everd_col_output_col_name] = (
        groupeddf[everd_weighted_col] / groupeddf[total_volume_col].replace(0, np.nan)
    )
    output_table = groupeddf[[everd_col_output_col_name, total_volume_col]]
    return output_table


def plot_everd_column(
    df,
    everdqco_col,
    everd_col_output_col_name,
    groupby_col,
    weight_col,
    trim_n = None,
    total_volume_col="Total Origination Volume",
    category_col=None,
    categories_to_plot=None,
    max_categories=None,
    other_label="Other",
    category_missing_label="Missing",
    title=None,
    x_label=None,
    value_ylabel=None,
    volume_ylabel="Total Origination Volume",
    figsize=(12, 6),
    volume_color="#d9e2ec",
    volume_edge_color="#8aa2b8",
    volume_alpha=0.45,
    marker="o",
    linewidth=2,
    right_colors=None,
    legend_location="upper left",
    value_y_min=None,
    value_y_max=None,
    volume_y_min=None,
    volume_y_max=None,
    value_y_formatter="percent",
    volume_display="background",
    title_fontsize=16,
    label_fontsize=13,
    tick_fontsize=12,
    legend_fontsize=12,
    show=True,
    return_results=False,
):

    x_axis_label = format_column_label(groupby_col) if x_label is None else x_label
    output_table = get_everd_co_table(
        df,
        everdqco_col=everdqco_col,
        everd_col_output_col_name=everd_col_output_col_name,
        groupby_col=groupby_col,
        weight_col=weight_col,
        total_volume_col=total_volume_col,
        category_col=category_col,
        categories_to_plot=categories_to_plot,
        max_categories=max_categories,
        other_label=other_label,
        category_missing_label=category_missing_label,
    )
    value_cols = [col for col in output_table.columns if col != total_volume_col]
    plot_title = title
    if plot_title is None:
        if category_col is not None:
            plot_title = (
                f"{everd_col_output_col_name} by {format_column_label(category_col)} vs {{volume}}"
            )
        else:
            plot_title = f"{everd_col_output_col_name} vs {{volume}}"

    if trim_n is not None and trim_n > 0:
        output_table.loc[
            output_table.index[-trim_n:],
            value_cols
        ] = np.nan

    charts = plot_metric_vs_volume(
        output_table,
        value_cols=value_cols,
        volume_col=total_volume_col,
        title=plot_title,
        x_label=x_axis_label,
        value_ylabel=value_ylabel or everd_col_output_col_name,
        volume_ylabel=volume_ylabel,
        figsize=figsize,
        volume_color=volume_color,
        volume_edge_color=volume_edge_color,
        volume_alpha=volume_alpha,
        marker=marker,
        linewidth=linewidth,
        right_colors=right_colors,
        legend_location=legend_location,
        value_y_min=value_y_min,
        value_y_max=value_y_max,
        volume_y_min=volume_y_min,
        volume_y_max=volume_y_max,
        value_y_formatter=value_y_formatter,
        volume_display=volume_display,
        format_value_labels=category_col is None,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        show=show,
    )

    if return_results:
        return {
            "table": output_table,
            "charts": charts,
        }

    return output_table


def add_normalized_interest_rate(
    df,
    rate_col="Booking_Interest_Rate",
    weight_col="Booked_Amount",
    vintage_col="vintage_quarter",
    output_col="normalized_interest_rate",
    benchmark_col="weighted_vintage_rate",
):
    out = df.copy()

    out[rate_col] = pd.to_numeric(out[rate_col], errors="coerce")
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce")

    valid = (
        out[rate_col].notna()
        & out[weight_col].notna()
        & (out[weight_col] > 0)
        & out[vintage_col].notna()
    )

    vintage_rates = (
        out.loc[valid]
        .assign(_rate_x_weight=lambda x: x[rate_col] * x[weight_col])
        .groupby(vintage_col, as_index=False)
        .agg(
            _rate_x_weight_sum=("_rate_x_weight", "sum"),
            _weight_sum=(weight_col, "sum"),
        )
    )

    vintage_rates[benchmark_col] = (
        vintage_rates["_rate_x_weight_sum"] / vintage_rates["_weight_sum"]
    )

    out = out.merge(
        vintage_rates[[vintage_col, benchmark_col]],
        on=vintage_col,
        how="left",
    )

    out[output_col] = out[rate_col] - out[benchmark_col]

    return out


def trim_last_n_values(df, cols, n=1, inplace=False):
    """
    Set the last n rows of selected columns to NaN.
    Useful for hiding incomplete recent periods in plots.
    """
    if n is None or n <= 0:
        return df if inplace else df.copy()

    out = df if inplace else df.copy()

    if isinstance(cols, str):
        cols = [cols]

    cols = [col for col in cols if col in out.columns]

    if not cols:
        return out

    out.loc[out.index[-n:], cols] = np.nan
    return out


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_one_visualization_decile_everdq_distribution(df, 
                                                      col,
                                                      everD_co_col,
                                                      orig_bal_col, 
                                                      breakpoints=None, 
                                                      n_bins=10,
                                                      top_n_categories=None,
                                                      other_label="Other",
                                                      category_order=None,
                                                      category_sort_ascending=False,
                                                      title=None,
                                                      y_label=None,
                                                      figsize=(11, 5)):

    display_col = col.replace("_Bucket", "").replace("_", " ")
    s = df[col]
    n_unique = s.nunique(dropna=True)
    is_categorical = (s.dtype == "object") or (s.dtype.name == "category")

    # ---------- CASE 1: categorical or <= 10 unique values ----------
    if breakpoints is None and (is_categorical or n_unique <= 10):
        plot_df = df.copy()
        plot_col = col

        if top_n_categories is not None:
            if top_n_categories < 1:
                raise ValueError("top_n_categories must be at least 1.")

            plot_col = f"_{col}_plot_category"
            top_categories = (
                plot_df.groupby(col, dropna=False)[orig_bal_col]
                .sum()
                .sort_values(ascending=False)
                .head(top_n_categories)
                .index
            )
            plot_df[plot_col] = plot_df[col].where(
                plot_df[col].isin(top_categories),
                other_label,
            )

        grouped = plot_df.groupby(plot_col, dropna=False, sort=False)
        total_bal = grouped[orig_bal_col].sum()
        bad_bal = (
            plot_df.loc[plot_df[everD_co_col] == 1]
            .groupby(plot_col, dropna=False)[orig_bal_col]
            .sum()
        )
        pct = bad_bal.reindex(total_bal.index, fill_value=0).div(total_bal).fillna(0)
        counts = grouped.size().reindex(pct.index)

        if category_order is not None:
            if isinstance(category_order, str):
                sort_key = category_order.lower()
                if sort_key in ["rate", "pct", "everd", "everdq"]:
                    ordered_index = pct.sort_values(ascending=category_sort_ascending).index
                elif sort_key in ["count", "n"]:
                    ordered_index = list(counts.sort_values(ascending=category_sort_ascending).index)
                    if other_label in ordered_index:
                        ordered_index = [value for value in ordered_index if value != other_label]
                        ordered_index.append(other_label)
                elif sort_key in ["balance", "bal", "volume", "orig_bal"]:
                    ordered_index = total_bal.sort_values(ascending=category_sort_ascending).index
                elif sort_key in ["alpha", "alphabetical", "name"]:
                    ordered_index = sorted(pct.index, key=lambda x: str(x))
                    if not category_sort_ascending:
                        ordered_index = list(reversed(ordered_index))
                else:
                    raise ValueError(
                        "category_order must be a list or one of: "
                        "'rate', 'count', 'balance', 'alpha'."
                    )
            else:
                manual_order = list(category_order)
                ordered_index = [value for value in manual_order if value in pct.index]
                ordered_index += [value for value in pct.index if value not in ordered_index]

            pct = pct.reindex(ordered_index)
            counts = counts.reindex(ordered_index)

        plt.figure(figsize=figsize)
        ax = pct.plot(kind="bar", color="#4C78A8", edgecolor="white", width=0.8)
        ymax = pct.max() if len(pct) else 0
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel(y_label or f"DQ60/CO% of {orig_bal_col}")
        ax.set_xlabel("")
        ax.set_title(title or f"DQ60/CO Rate by {display_col}", fontsize=13, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 0.05)

        # Add sample size above bars
        for i, value in enumerate(pct):
            ax.text(
                i,
                value + (ymax * 0.03 if ymax > 0 else 0.002),
                f"{value:.1%}\nn={counts.iloc[i]:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # ---------- CASE 2: numeric with > 10 values ----------
    else:
        numeric_s = pd.to_numeric(s, errors="coerce")
        valid = df[numeric_s.notna()].copy()
        numeric_s = numeric_s[numeric_s.notna()]

        if numeric_s.nunique() == 0:
            return

        if breakpoints is not None:
            data_min = numeric_s.min()
            data_max = numeric_s.max()
            breakpoint_values = (
                pd.Series(breakpoints, dtype="float64")
                .dropna()
                .sort_values()
                .unique()
            )
            breakpoint_values = [
                value for value in breakpoint_values
                if data_min < value < data_max
            ]
            cut_bins = [data_min] + breakpoint_values + [data_max]
            bins = pd.cut(numeric_s, bins=cut_bins, include_lowest=True)
        else:
            try:
                bins = pd.qcut(numeric_s, q=n_bins, duplicates="drop")
            except ValueError:
                bins = pd.cut(numeric_s, bins=n_bins)

        valid["_bin"] = bins

        grouped = valid.groupby("_bin", dropna=False)
        total_bal = grouped[orig_bal_col].sum()
        bad_bal = (
            valid.loc[valid[everD_co_col] == 1]
            .groupby("_bin", dropna=False)[orig_bal_col]
            .sum()
        )
        pct = bad_bal.reindex(total_bal.index, fill_value=0).div(total_bal).fillna(0)
        counts = grouped.size().reindex(pct.index)

        plt.figure(figsize=figsize)
        ax = pct.plot(kind="bar", color="#4C78A8", edgecolor="white", width=0.8)
        ymax = pct.max() if len(pct) else 0
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel(y_label or f"DQ60/CO% of {orig_bal_col}")
        ax.set_xlabel("")
        ax.set_title(title or f"DQ60/CO Rate by {display_col}", fontsize=13, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 0.05)

        # Add sample size text
        for i, value in enumerate(pct):
            ax.text(
                i,
                value + (ymax * 0.03 if ymax > 0 else 0.002),
                f"{value:.1%}\nn={counts.iloc[i]:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
