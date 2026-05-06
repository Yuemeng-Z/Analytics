import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

from scipy.stats import gaussian_kde

# ---------- Formatters ----------

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


def trim_last_n(col, n=3):
    # mask = col.notna() & (col != 0)
    mask = col.notna()
    idx = col.index[mask]
    col.loc[idx[-n:]] = float('nan')  # works even if fewer than n
    return col


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
        wt_distribution = wt_distribution.applymap(lambda x: f"{x:.2%}")
    if show_total_origination:
        wt_distribution.loc['Total Origination'] = np.array(wt_distribution_.sum(axis=0))
    
    return wt_distribution


# def weighted_avg(df, cols, weight_col):
#     w = df[weight_col]
#     return df[cols].apply(lambda x: np.average(x, weights=w))

def weighted_avg(series, weights):
    mask = series.notna() & weights.notna() & (series > 0) & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(series[mask], weights=weights[mask])

def groupby_weighted_avg(df, groupby_col, cols, weight_col):

    result = (
    df.groupby(groupby_col)
      .apply(lambda g: pd.Series({
          col: weighted_avg(g[col], g[weight_col])
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
                        legend_location = "upper left"

                       ):
    plt.style.use("default")

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
            df.index,
            df[col],
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
                    loan_id_col_name):
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
        df_cut.assign(co_flag=lambda x: x["Default"] > 0)
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
                             dpd_num = 30, 
                             mob_col_name = 'MOB', 
                             loan_id_col_name = "LoanID",
                             original_bal_col_name = 'AmountFinanced',
                             vintage_col_name = 'FundingYearMonth',
                             mob_orig_data = 1
                             ):
    ever_dq_res = calc_everd30_co(df, age_cutoff, dpd_col_name, dpd_num, mob_col_name, loan_id_col_name)
    ever_dq_co = df[df[mob_col_name] == mob_orig_data].merge(ever_dq_res, on = loan_id_col_name, how = 'left')
    ever_dq_co['EverD{:.0f} @ MOB{:.0f}'.format(dpd_num, age_cutoff)] = ever_dq_co['everD30 or CO'] * ever_dq_co[original_bal_col_name]
    out = ever_dq_co[[original_bal_col_name, vintage_col_name] + ['EverD{:.0f} @ MOB{:.0f}'.format(dpd_num, age_cutoff)]].groupby(vintage_col_name).sum()
    out['EverD{:.0f}% @ MOB{:.0f}'.format(dpd_num, age_cutoff)] = out['EverD{:.0f} @ MOB{:.0f}'.format(dpd_num, age_cutoff)] / out[original_bal_col_name]

    return out, ever_dq_co
