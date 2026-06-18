"""
Core loan data pipeline helpers.

Put reusable file reading, standardization, validation, merging, panel shaping,
target creation, and loan performance rollups here. These helpers should stay
quiet: return DataFrames/objects without notebook display side effects.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FICO_BINS = [-np.inf, 580, 620, 660, 700, 740, 780, 820, np.inf]
DEFAULT_FICO_LABELS = [
    "<580",
    "580-619",
    "620-659",
    "660-699",
    "700-739",
    "740-779",
    "780-819",
    "820+",
]


def read_tabular(path, **kwargs):
    path = Path(path).expanduser()
    suffix = path.suffix.lower()

    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path, **kwargs)
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, **kwargs)

    raise ValueError(f"Unsupported file type: {suffix}")


def standardize_columns(df, column_map):
    rename_map = {
        source_col: standard_col
        for standard_col, source_col in column_map.items()
        if source_col and source_col in df.columns
    }
    return df.rename(columns=rename_map).copy()


def missing_columns(df, required_cols):
    return [col for col in required_cols if col not in df.columns]


def coerce_datetime(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def coerce_numeric(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_fico_bucket(
    df,
    fico_col="fico",
    bucket_col="fico_bucket",
    bins=None,
    labels=None,
    missing_label="Missing",
):
    out = df.copy()
    bins = DEFAULT_FICO_BINS if bins is None else bins
    labels = DEFAULT_FICO_LABELS if labels is None else labels

    bucket = pd.cut(out[fico_col], bins=bins, labels=labels, right=False)
    out[bucket_col] = bucket.astype("object")
    out.loc[out[fico_col].isna(), bucket_col] = missing_label
    return out


def weighted_average(series, weights):
    mask = series.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(series[mask], weights=weights[mask])


def weighted_missing_rate(series, weights):
    mask = weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return weights[mask & series.isna()].sum() / weights[mask].sum()


def trim_last_n(col, n=3):
    mask = col.notna()
    idx = col.index[mask]
    col.loc[idx[-n:]] = float("nan")
    return col


def data_quality_report(df, required_cols=None, weight_col=None):
    required_cols = [] if required_cols is None else required_cols
    if len(df.columns) == 0:
        return pd.DataFrame(
            columns=[
                "column",
                "dtype",
                "required",
                "non_null",
                "missing",
                "missing_pct",
                "unique",
                "weighted_missing_pct",
            ]
        )

    weight = df[weight_col] if weight_col in df.columns else None

    rows = []
    for col in df.columns:
        row = {
            "column": col,
            "dtype": str(df[col].dtype),
            "required": col in required_cols,
            "non_null": df[col].notna().sum(),
            "missing": df[col].isna().sum(),
            "missing_pct": df[col].isna().mean(),
            "unique": df[col].nunique(dropna=True),
        }
        if weight is not None:
            row["weighted_missing_pct"] = weighted_missing_rate(df[col], weight)
        rows.append(row)

    report = pd.DataFrame(rows)
    return report.sort_values(["required", "missing_pct"], ascending=[False, False])


def weighted_summary(df, groupby_cols=None, weight_col="original_balance", numeric_cols=None):
    numeric_cols = [] if numeric_cols is None else numeric_cols
    groupby_cols = [] if groupby_cols is None else groupby_cols

    if not groupby_cols:
        grouped = [("All", df)]
    else:
        grouped = df.groupby(groupby_cols, dropna=False)

    rows = []
    for key, group in grouped:
        row = {}
        if groupby_cols:
            key = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(groupby_cols, key)))
        else:
            row["segment"] = key

        row["loan_count"] = group["loan_id"].nunique() if "loan_id" in group.columns else len(group)
        row["record_count"] = len(group)

        if weight_col in group.columns:
            weight = group[weight_col]
            row[f"{weight_col}_sum"] = weight.sum()
            for col in numeric_cols:
                if col in group.columns:
                    row[f"wa_{col}"] = weighted_average(group[col], weight)
                    row[f"missing_{col}_pct"] = group[col].isna().mean()
                    row[f"weighted_missing_{col}_pct"] = weighted_missing_rate(group[col], weight)

        rows.append(row)

    return pd.DataFrame(rows)


def weighted_distribution(
    df,
    category_col,
    weight_col="original_balance",
    by_col=None,
    normalize=True,
    include_missing=True,
    missing_label="Missing",
):
    work = df.copy()
    if include_missing:
        work[category_col] = work[category_col].astype("object").where(
            work[category_col].notna(),
            missing_label,
        )

    if by_col:
        table = work.pivot_table(
            values=weight_col,
            index=category_col,
            columns=by_col,
            aggfunc="sum",
            fill_value=0,
        )
        if normalize:
            denom = table.sum(axis=0).replace(0, np.nan)
            table = table.div(denom, axis=1)
        return table

    table = work.groupby(category_col, dropna=False)[weight_col].sum().to_frame("balance")
    if normalize:
        total = table["balance"].sum()
        table["pct"] = table["balance"] / total if total else np.nan
    return table.sort_values("balance", ascending=False)


def merge_coverage_report(origination, performance, loan_id_col="loan_id"):
    orig_ids = pd.Index(origination[loan_id_col].dropna().unique())
    perf_ids = pd.Index(performance[loan_id_col].dropna().unique())

    rows = [
        {"metric": "origination_unique_loans", "value": len(orig_ids)},
        {"metric": "performance_unique_loans", "value": len(perf_ids)},
        {"metric": "performance_loans_missing_origination", "value": len(perf_ids.difference(orig_ids))},
        {"metric": "origination_loans_missing_performance", "value": len(orig_ids.difference(perf_ids))},
    ]
    return pd.DataFrame(rows)


def merge_origination_performance(
    origination,
    performance,
    loan_id_col="loan_id",
    how="left",
    suffixes=("", "_orig"),
):
    return performance.merge(
        origination,
        on=loan_id_col,
        how=how,
        suffixes=suffixes,
    )


def first_observation_by_loan(df, loan_id_col="loan_id", sort_cols=None):
    sort_cols = [] if sort_cols is None else sort_cols
    if loan_id_col not in df.columns:
        return df.copy()

    present_sort_cols = [col for col in sort_cols if col in df.columns]

    work = df.copy()
    if present_sort_cols:
        work = work.sort_values([loan_id_col] + present_sort_cols)

    return work.drop_duplicates(subset=[loan_id_col], keep="first").copy()


def origination_from_panel(df, loan_id_col="loan_id", sort_cols=None, keep_cols=None):
    origination = first_observation_by_loan(
        df,
        loan_id_col=loan_id_col,
        sort_cols=sort_cols,
    )

    if keep_cols is None:
        return origination

    present_cols = [col for col in keep_cols if col in origination.columns]
    return origination[present_cols].copy()


def performance_from_panel(df, keep_cols=None):
    if keep_cols is None:
        return df.copy()

    present_cols = [col for col in keep_cols if col in df.columns]
    return df[present_cols].copy()


def add_vintage_and_mob(
    df,
    origination_date_col="origination_date",
    performance_date_col="performance_date",
    vintage_col="vintage",
    mob_col="mob",
    vintage_freq="Q",
):
    out = df.copy()
    out[origination_date_col] = pd.to_datetime(out[origination_date_col], errors="coerce")
    out[performance_date_col] = pd.to_datetime(out[performance_date_col], errors="coerce")
    out[vintage_col] = out[origination_date_col].dt.to_period(vintage_freq).astype("string")

    orig_period = out[origination_date_col].dt.to_period("M")
    perf_period = out[performance_date_col].dt.to_period("M")
    out[mob_col] = [
        (perf - orig).n if pd.notna(orig) and pd.notna(perf) else np.nan
        for orig, perf in zip(orig_period, perf_period)
    ]
    return out


def panel_coverage_report(
    df,
    loan_id_col="loan_id",
    performance_date_col="performance_date",
    mob_col="mob",
):
    metrics = [
        {"metric": "record_count", "value": len(df)},
        {
            "metric": "unique_loans",
            "value": df[loan_id_col].nunique(dropna=True) if loan_id_col in df.columns else np.nan,
        },
    ]

    if performance_date_col in df.columns:
        metrics.extend([
            {"metric": "min_performance_date", "value": df[performance_date_col].min()},
            {"metric": "max_performance_date", "value": df[performance_date_col].max()},
        ])

    if mob_col in df.columns:
        metrics.extend([
            {"metric": "min_mob", "value": df[mob_col].min()},
            {"metric": "max_mob", "value": df[mob_col].max()},
        ])

    return pd.DataFrame(metrics)


def ever_dq_target_from_panel(
    df,
    loan_id_col="loan_id",
    mob_col="mob",
    days_delinquent_col="days_delinquent",
    chargeoff_amount_col=None,
    cutoff_mob=12,
    dq_threshold=60,
    target_col=None,
):
    target_col = target_col or f"ever_dq{dq_threshold}_or_co_mob{cutoff_mob}"

    required_cols = [loan_id_col, mob_col, days_delinquent_col]
    missing = missing_columns(df, required_cols)
    if missing:
        raise ValueError(f"Missing required columns for target creation: {missing}")

    df_cut = df[df[mob_col] <= cutoff_mob].copy()
    dq_flag = (
        df_cut.assign(dq_flag=lambda x: x[days_delinquent_col] >= dq_threshold)
        .groupby(loan_id_col)["dq_flag"]
        .max()
        .rename(f"ever_dq{dq_threshold}_mob{cutoff_mob}")
    )

    pieces = [dq_flag]
    flag_cols = [dq_flag.name]

    if chargeoff_amount_col and chargeoff_amount_col in df_cut.columns:
        co_flag = (
            df_cut.assign(co_flag=lambda x: x[chargeoff_amount_col].fillna(0) > 0)
            .groupby(loan_id_col)["co_flag"]
            .max()
            .rename(f"ever_co_mob{cutoff_mob}")
        )
        pieces.append(co_flag)
        flag_cols.append(co_flag.name)

    target = pd.concat(pieces, axis=1).fillna(False)
    target[target_col] = target[flag_cols].any(axis=1).astype(int)

    return target.reset_index()


def _prediction_frame_with_reference(
    predictions,
    reference_df,
    id_col,
    reference_cols,
    sample_label,
):
    if predictions is None or len(predictions) == 0:
        return pd.DataFrame()

    out = predictions.copy()
    out["sample"] = sample_label

    if (
        reference_df is None
        or len(reference_df) == 0
        or id_col not in out.columns
        or id_col not in reference_df.columns
    ):
        return out

    reference_cols = [
        col
        for col in reference_cols
        if col in reference_df.columns and col not in [id_col, "y_true", "y_pred"]
    ]
    if not reference_cols:
        return out

    ref = first_observation_by_loan(reference_df, loan_id_col=id_col)
    ref = ref[[id_col] + reference_cols].copy()

    out = out.drop(columns=[col for col in reference_cols if col in out.columns])
    return out.merge(ref, on=id_col, how="left")


def build_regression_prediction_frame(
    results,
    reference_df,
    id_col="loan_id",
    reference_cols=None,
    test_reference_df=None,
):
    reference_cols = [] if reference_cols is None else reference_cols

    train = _prediction_frame_with_reference(
        results.get("X_train_with_preds"),
        reference_df,
        id_col=id_col,
        reference_cols=reference_cols,
        sample_label="Train",
    )

    test = _prediction_frame_with_reference(
        results.get("X_test_with_preds"),
        test_reference_df if test_reference_df is not None else reference_df,
        id_col=id_col,
        reference_cols=reference_cols,
        sample_label="Test",
    )

    return pd.concat([train, test], ignore_index=True)


def _weighted_rate(group, value_col, weight_col):
    mask = group[value_col].notna() & group[weight_col].notna() & (group[weight_col] > 0)
    if not mask.any():
        return np.nan
    return (group.loc[mask, value_col] * group.loc[mask, weight_col]).sum() / group.loc[mask, weight_col].sum()


def _factor_label(config):
    label = config.get("label")
    if label:
        return label

    kind = config.get("kind", "weighted_average")
    column = config.get("column")
    if kind in ["category_mix", "category_pct"]:
        return f"{column} = {config.get('category')}"
    if kind in ["weighted_missing_rate", "missing_rate"]:
        return f"{column} Missing %"
    return column


def _factor_by_group(reference_df, group_col, weight_col, config):
    column = config.get("column")
    label = _factor_label(config)
    kind = config.get("kind", "weighted_average")

    if reference_df is None or reference_df.empty:
        return pd.Series(dtype=float, name=label)
    if group_col not in reference_df.columns or column not in reference_df.columns:
        return pd.Series(dtype=float, name=label)

    rows = []
    for group_value, group in reference_df.groupby(group_col, dropna=False):
        weights = group[weight_col] if weight_col in group.columns else pd.Series(1, index=group.index)

        if kind in ["weighted_average", "wa"]:
            value = weighted_average(group[column], weights)
        elif kind in ["category_mix", "category_pct"]:
            category = config.get("category")
            mask = group[column].fillna(config.get("missing_label", "Missing")) == category
            valid_weight = weights.notna() & (weights > 0)
            value = weights[valid_weight & mask].sum() / weights[valid_weight].sum() if valid_weight.any() else np.nan
        elif kind in ["weighted_missing_rate", "missing_rate"]:
            value = weighted_missing_rate(group[column], weights)
        elif kind == "mean":
            value = group[column].mean()
        else:
            raise ValueError(f"Unsupported post-regression chart kind: {kind}")

        rows.append((group_value, value))

    return pd.Series(dict(rows), name=label)


def build_post_regression_review_table(
    results,
    reference_df,
    group_col,
    weight_col,
    chart_configs=None,
    id_col="loan_id",
    factor_reference_df=None,
    test_reference_df=None,
    factor_join="left",
):
    chart_configs = [] if chart_configs is None else chart_configs
    factor_reference_df = reference_df if factor_reference_df is None else factor_reference_df

    reference_cols = [group_col, weight_col]
    for config in chart_configs:
        if config.get("column"):
            reference_cols.append(config["column"])

    prediction_frame = build_regression_prediction_frame(
        results,
        reference_df=reference_df,
        id_col=id_col,
        reference_cols=reference_cols,
        test_reference_df=test_reference_df,
    )

    required_cols = ["sample", group_col, weight_col, "y_true", "y_pred"]
    if prediction_frame.empty or missing_columns(prediction_frame, required_cols):
        return pd.DataFrame()

    rows = []
    for (sample, group_value), group in prediction_frame.groupby(["sample", group_col], dropna=False):
        rows.append({
            group_col: group_value,
            "sample": sample,
            "Actual": _weighted_rate(group, "y_true", weight_col),
            "Model": _weighted_rate(group, "y_pred", weight_col),
            "weight": group[weight_col].sum(),
            "loan_count": group[id_col].nunique() if id_col in group.columns else len(group),
        })

    long_table = pd.DataFrame(rows)
    if long_table.empty:
        return long_table

    rate_table = long_table.pivot(index=group_col, columns="sample", values=["Actual", "Model"])
    rate_table.columns = [f"{metric} - {sample}" for metric, sample in rate_table.columns]

    support_table = long_table.pivot(index=group_col, columns="sample", values=["weight", "loan_count"])
    support_table.columns = [f"{metric} - {sample}" for metric, sample in support_table.columns]

    review_table = rate_table.join(support_table, how="outer")

    for config in chart_configs:
        factor = _factor_by_group(
            factor_reference_df,
            group_col=group_col,
            weight_col=weight_col,
            config=config,
        )
        review_table = review_table.join(factor, how=factor_join)

    return review_table.sort_index()


def plot_post_regression_review_charts(
    review_table,
    chart_configs,
    plot_func,
    outcome_label,
    title_prefix="Actual vs Model",
    data_labels=False,
):
    if review_table is None or review_table.empty or plot_func is None:
        return {}

    base_cols = [
        col
        for col in ["Actual - Train", "Model - Train", "Actual - Test", "Model - Test"]
        if col in review_table.columns
    ]
    chart_tables = {}

    for config in chart_configs:
        label = _factor_label(config)
        chart_cols = base_cols + ([label] if label in review_table.columns else [])
        if not chart_cols:
            continue

        output_plot = review_table[chart_cols].copy()
        secondary_y_cols = [label] if label in output_plot.columns else []
        secondary_percentage = config.get("kind") in ["category_mix", "category_pct", "weighted_missing_rate", "missing_rate"]

        plot_func(
            output_plot,
            title=f"{title_prefix} vs {label}",
            ylabel=outcome_label,
            percentage=True,
            data_labels=data_labels,
            secondary_y_cols=secondary_y_cols,
            secondary_ylabel=label if secondary_y_cols else "",
            secondary_percentage=secondary_percentage,
            use_custom_colors=False,
            linestyles_inputs=["-", "--", "-", "--", ":"],
        )
        chart_tables[label] = output_plot

    return chart_tables


def performance_rollup(
    df,
    vintage_col="vintage",
    mob_col="mob",
    original_balance_col="original_balance",
    current_balance_col="current_balance",
    delinquency_days_col="days_delinquent",
    chargeoff_amount_col=None,
    recovery_amount_col=None,
):
    required_cols = [
        "loan_id",
        vintage_col,
        mob_col,
        original_balance_col,
        current_balance_col,
    ]
    if missing_columns(df, required_cols):
        return pd.DataFrame()

    agg = {
        "loan_id": "nunique",
        original_balance_col: "sum",
        current_balance_col: "sum",
    }

    if chargeoff_amount_col and chargeoff_amount_col in df.columns:
        agg[chargeoff_amount_col] = "sum"
    if recovery_amount_col and recovery_amount_col in df.columns:
        agg[recovery_amount_col] = "sum"

    grouped = df.groupby([vintage_col, mob_col], dropna=False).agg(agg).reset_index()
    grouped = grouped.rename(
        columns={
            "loan_id": "loan_count",
            original_balance_col: "original_balance",
            current_balance_col: "current_balance",
        }
    )

    if delinquency_days_col in df.columns:
        dq = df.assign(
            dq30_balance=np.where(df[delinquency_days_col] >= 30, df[current_balance_col], 0)
        )
        dq = dq.groupby([vintage_col, mob_col], dropna=False)["dq30_balance"].sum().reset_index()
        grouped = grouped.merge(dq, on=[vintage_col, mob_col], how="left")
        grouped["dq30_pct"] = grouped["dq30_balance"] / grouped["current_balance"]

    return grouped


def create_vintage(origination_raw, origination_date_col):
    origination_raw[origination_date_col] = pd.to_datetime(origination_raw[origination_date_col], errors='coerce')
    origination_raw['Origination Month'] = origination_raw[origination_date_col].dt.to_period('M')
    origination_raw['Origination Quarter'] = origination_raw[origination_date_col].dt.to_period('Q')

    return origination_raw
