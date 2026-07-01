"""
Regression and model-review helpers.

Keep sklearn transformers, model training, model diagnostics, coefficient
summaries, category reduction, and regression plotting here. Data reading and
loan tape preparation belong in loanPipelineHelpers.py.
"""

import numpy as np
import pandas as pd

# Sklearn core
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Model selection
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    r2_score
)

# Stats
from parentHelpers import get_wt_distribution, groupby_weighted_avg, plot_correlation_simple
from scipy import stats

# Plotting
import matplotlib.pyplot as plt

# =========================
# SAS HAT SPLINE TRANSFORMER
# =========================
class SASHatSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, breaks_dict, drop_features=None):
        self.breaks_dict = breaks_dict
        self.drop_features = drop_features or {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # Track which variables actually have missing values
        self.has_missing_ = {}
        for var in self.breaks_dict:
            self.has_missing_[var] = X[var].isna().any()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        # print(self.breaks_dict.items())
        for var, breaks in self.breaks_dict.items():
            x = X[var]
            # Only add missing column if needed
            if self.has_missing_[var]:
                out[f"{var}_missing_0"] = x.isna().astype(int)

            nan_mask = x.notna()
            x_filled = x.fillna(0)
            knots = [-np.inf] + breaks + [np.inf]

            for i in range(1, len(knots) - 1):
                left = knots[i - 1]
                center = knots[i]
                right = knots[i + 1]

                col = f"{var}_{_format_break(center)}_{(i):.0f}"
                out[col] = 0.0

                mask = (x_filled > left) & (x_filled <= center) & nan_mask
                out.loc[mask, col] = (x_filled[mask] - left) / (center - left)

                mask = (x_filled > center) & (x_filled <= right) & nan_mask
                out.loc[mask, col] = (right - x_filled[mask]) / (right - center)

                if np.isinf(left):
                    out.loc[(x_filled <= center) & (nan_mask), col] = 1.0

                if np.isinf(right):
                    out.loc[(x_filled > center) & (nan_mask), col] = 1.0

            if var in self.drop_features:
                drop_col = self.drop_features[var]
                if drop_col in out.columns:   
                    out.drop(columns=[drop_col], inplace=True)

        return out

    def get_feature_names_out(self, input_features=None):
        feature_names = []

        for var, breaks in self.breaks_dict.items():
            add_missing = self.has_missing_.get(var, False)

            # Missing indicator
            if add_missing:
                feature_names.append(f"{var}_missing_0")

            knots = [-np.inf] + breaks + [np.inf]

            for i in range(1, len(knots) - 1):
                center = knots[i]
                col = f"{var}_{_format_break(center)}_{(i):.0f}"
                feature_names.append(col)

            # Drop feature if specified
            if var in self.drop_features:
                drop_col = self.drop_features[var]
                if drop_col in feature_names:
                    feature_names.remove(drop_col)

        return np.array(feature_names)
    
# =========================
# Compute stats (p-values, confidence intervals) for sklearn models
# =========================
def compute_sklearn_stats(pipeline, X, y, model_type="linear"):

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(raw_feature_names)

    X_design = preprocessor.transform(X)
    X_design = np.asarray(X_design)

    X_design_int = np.column_stack([np.ones(X_design.shape[0]), X_design])

    n, p = X_design_int.shape

    if model_type == "linear":

        y_pred = pipeline.predict(X)
        residuals = y - y_pred

        sigma2 = (residuals @ residuals) / (n - p)

        cov = sigma2 * np.linalg.pinv(X_design_int.T @ X_design_int)

        se = np.sqrt(np.diag(cov))

        coef = np.concatenate(([model.intercept_], model.coef_))

        stat = coef / se

        p_values = 2 * (1 - stats.t.cdf(np.abs(stat), df=n - p))

        crit = stats.t.ppf(0.975, df=n - p)
        stat_name = 'T-'

    else:

        probs = pipeline.predict_proba(X)[:, 1]

        eps = 1e-6
        p = np.clip(probs, eps, 1 - eps)
        weights = p * (1 - p)

        cov = np.linalg.pinv(X_design_int.T @ (X_design_int * weights[:, None]))

        se = np.sqrt(np.diag(cov))

        coef = np.concatenate(([model.intercept_[0]], model.coef_.ravel()))

        stat = coef / se

        p_values = 2 * (1 - stats.norm.cdf(np.abs(stat)))

        crit = stats.norm.ppf(0.975)
        stat_name = 'Z-'

    ci_lower = coef - crit * se
    ci_upper = coef + crit * se

    stats_df = pd.DataFrame({
        "Feature": ["Intercept"] + list(feature_names),
        "Coefficient": coef,
        "Std Error": se,
        stat_name + "Stat": stat,
        "P Value": p_values,
        "CI lower": ci_lower,
        "CI upper": ci_upper
    })

    return stats_df


# =========================
# Helper Functions
# =========================
def clean_numeric_data(df, numeric_features):
    df = df.copy()

    for col in numeric_features:
        # Replace inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Fill NaN (choose your strategy)
        df[col] = df[col].fillna(df[col].median())

    return df

def clean_feature_names(feature_names):
    cleaned = []

    for f in feature_names:
        # Remove ColumnTransformer prefixes
        if "__" in f:
            f = f.split("__", 1)[1]

        cleaned.append(f)

    return cleaned

def sort_spline_by_var_legacy(df):
    df = df.copy()

    def get_var(f):
        if f in ["Intercept", "R²", "Accuracy", "AUC"]:
            return "zzz_summary"
        return f.split("_")[0]
    
    def get_order(f):
        if f in ["Intercept", "R²", "Accuracy", "AUC"]:
            return "zzz_summary"
        return f.split("_")[-1]

    df["_var"] = df["Feature"].apply(get_var)

    # Preserve original order within each variable using index
    df["_orig_order"] = df["Feature"].apply(get_order)

    df = df.sort_values(["_var", "_orig_order"]).drop(columns=["_var", "_orig_order"])

    return df

def sort_spline_by_var(df, spline_features=None):
    df = df.copy()
    summary_features = {"Intercept", "R²", "Accuracy", "AUC"}

    if isinstance(spline_features, dict):
        spline_vars = list(spline_features.keys())
    elif spline_features:
        spline_vars = list(spline_features)
    else:
        spline_vars = []

    spline_order = {var: i for i, var in enumerate(spline_vars)}
    spline_vars_by_length = sorted(spline_vars, key=len, reverse=True)

    def get_spline_var(feature):
        if not isinstance(feature, str):
            return None

        for var in spline_vars_by_length:
            if feature.startswith(f"{var}_"):
                return var

        return None

    def get_spline_order(feature):
        try:
            return int(str(feature).rsplit("_", 1)[-1])
        except ValueError:
            return 0

    df["_orig_order"] = range(len(df))
    df["_spline_var"] = df["Feature"].apply(get_spline_var)
    df["_is_summary"] = df["Feature"].isin(summary_features)

    df["_section"] = 0
    df.loc[df["_spline_var"].notna(), "_section"] = 1
    df.loc[df["_is_summary"], "_section"] = 2

    df["_var_order"] = df.apply(
        lambda row: spline_order.get(row["_spline_var"], row["_orig_order"]),
        axis=1
    )
    df["_feature_order"] = df.apply(
        lambda row: get_spline_order(row["Feature"]) if row["_spline_var"] else row["_orig_order"],
        axis=1
    )

    df = df.sort_values(
        ["_section", "_var_order", "_feature_order", "_orig_order"]
    ).drop(
        columns=[
            "_orig_order",
            "_spline_var",
            "_is_summary",
            "_section",
            "_var_order",
            "_feature_order"
        ]
    )

    return df

# =========================
# AUTO BASELINE SELECTION
# =========================

def find_middle_risk_baseline(X, y, col):

    df = pd.DataFrame({col: X[col], "target": y})

    stats = df.groupby(col).agg(
        mean_target=("target","mean"),
        count=("target","size")
    )

    stats = stats.sort_values("mean_target")

    cum = stats["count"].cumsum()
    total = stats["count"].sum()

    baseline = stats.index[(cum >= total/2)][0]

    return baseline


def find_middle_spline_baseline(X, var, breaks):
    """
    Always pick the middle breakpoint (by index).
    """

    if len(breaks) == 0:
        raise ValueError(f"No breaks provided for {var}")

    mid_idx = len(breaks) // 2
    mid_value = breaks[mid_idx]

    return f"{var}_{_format_break(mid_value)}_{(mid_idx+1):.0f}"


def find_manual_spline_baseline(var, breaks, manual_value):
    """
    Resolve a user-provided spline baseline to the generated spline column name.
    manual_value can be either the breakpoint value or the exact spline feature name.
    """
    available_features = [
        f"{var}_{_format_break(value)}_{idx:.0f}"
        for idx, value in enumerate(breaks, start=1)
    ]

    if manual_value in available_features:
        return manual_value

    try:
        manual_numeric = float(manual_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Spline baseline '{manual_value}' for {var} is not valid. "
            f"Use one of these breakpoints: {breaks}, or one of these feature names: "
            f"{available_features}"
        )

    for idx, value in enumerate(breaks, start=1):
        if np.isclose(manual_numeric, float(value)):
            return f"{var}_{_format_break(value)}_{idx:.0f}"

    raise ValueError(
        f"Spline baseline '{manual_value}' for {var} is not one of the spline "
        f"breakpoints. Available breakpoints: {breaks}"
    )

# =========================
# FEATURE IMPORTANCE PLOT
# =========================
def plot_feature_importance(coef_df, top_n=15):
    df = coef_df.copy()

    # Keep only real features (exclude summary rows)
    df = df[
        ~df["Feature"].isin(["Intercept", "R²", "Accuracy", "AUC"])
    ]

    # Also drop any rows where coefficient is NaN (extra safety)
    df = df[df["Coefficient"].notna()]

    # Rank by absolute importance
    df["abs_coef"] = df["Coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_n)

    # Plot
    plt.close('all')
    plt.figure()
    plt.barh(df["Feature"], df["Coefficient"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance (by coefficient)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()


def create_spline_features_inputs(df, 
                                  piecewise_features,
                                  percentiles=[10,30,50,70,90]):
    # Create a dictionary to store percentiles for each variable
    percentile_dict = {}

    for column in piecewise_features:
        percentiles_for_variable = []
        for percentile in percentiles:
            value = np.percentile(df[column], percentile)
            #round value to 2 decimal places
            value = round(value, 2)
            percentiles_for_variable.append(value)
        percentile_dict[column] = percentiles_for_variable

    return percentile_dict


# =========================
# TRAIN MODEL
# =========================
def train_model(
                X_train,
                y_train,
                categorical_features,
                numeric_features,
                spline_features=None,
                manual_baseline=None,
                spline_manual_baseline=None,
                model_type="linear"
                ):
    
    spline_vars = list(spline_features.keys()) if spline_features else []

    transformers = []

    # categorical
    # =========================
    # AUTO BASELINE (CAT)
    # =========================

    manual_baseline = manual_baseline or {}
    spline_manual_baseline = spline_manual_baseline or {}

    cat_categories = []
    cat_drop = []

    for col in categorical_features:

        categories = list(X_train[col].dropna().astype("category").cat.categories)

        # ensure "Other" included if exists
        if "Other" in X_train[col].values and "Other" not in categories:
            categories.append("Other")

        if col in manual_baseline:
            best = manual_baseline[col]
            # print(f"[CAT] {col} baseline (manual) → {best}")
        else:
            best = find_middle_risk_baseline(X_train, y_train, col)
            # print(f"[CAT] {col} baseline (auto) → {best}")

        # critical safety check
        if best not in categories:
            raise ValueError(
                f"Baseline '{best}' not in categories for {col}. "
                f"Available: {categories}"
            )

        cat_categories.append(categories)
        cat_drop.append(best)


    # print("cat_categories:", cat_categories)
    # print("cat_drop:", cat_drop)
    # print("num categorical cols:", len(categorical_features))

    encoder = OneHotEncoder(
        categories=cat_categories,
        drop=cat_drop,
        sparse_output=False
    )

    transformers.append(("cat", encoder, categorical_features))

    # regular numeric
    if len(numeric_features) > 0:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )

    # spline numeric
    spline_drop = {}

    if spline_features:

        for var, breaks in spline_features.items():
            # best = find_middle_risk_spline_baseline(X_train, y_train, var, breaks)
            if var in spline_manual_baseline:
                best = find_manual_spline_baseline(
                    var,
                    breaks,
                    spline_manual_baseline[var]
                )
            else:
                best = find_middle_spline_baseline(X_train, var, breaks)

            # print(f"[SPLINE] {var} baseline → {best}")

            spline_drop[var] = best

        transformers.append(
            ("spline", SASHatSplineTransformer(spline_features, spline_drop), spline_vars)
        )

    preprocessor = ColumnTransformer(transformers)

    if model_type == "linear":
        model = LinearRegression()

    elif model_type == "logistic":
        if pd.Series(y_train).nunique() > 2:
            raise ValueError("Logistic regression requires binary target")
        model = LogisticRegression(
            # penalty="l2",
            # class_weight="balanced",
            penalty="l2",
            C=1.0,
            solver='liblinear',
            max_iter=1000
        )

    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X = X_train[categorical_features + numeric_features + spline_vars]
    y = y_train

    pipeline.fit(X, y)

    return pipeline, X, y, spline_drop

def _format_break(x):
    return str(int(x)) if float(x).is_integer() else str(x)

# =========================
# SUMMARIZE MODEL
# =========================
def summarize_model(pipeline, X, y, categorical_features, numeric_features,
                    spline_features=None, spline_drop=None,
                    model_type="linear", output_coef=False,
                    output_file_name='Regression_Coefficients.xlsx',
                    plot=True,
                    plot_top_n_features = 15,
                    adjust_logistic_decision_threshold=None,
                    intercept_multiplier=1.0):  

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(raw_feature_names)
    coefs = model.coef_.ravel()

    # =========================
    # REBUILD COEF TABLE (WITH BASELINE)
    # =========================
    cat_transformer = preprocessor.named_transformers_["cat"]

    if hasattr(cat_transformer, "named_steps"):
        encoder = cat_transformer.named_steps["onehot"]
        reducer = cat_transformer.named_steps.get("reduce", None)
    else:
        encoder = cat_transformer
        reducer = None

    X_cat = pd.DataFrame(X[categorical_features]).copy()

    if reducer is not None:
        X_cat = reducer.transform(X_cat)

    coef_iter = iter(coefs)
    rows = []

    # # =========================
    # # CATEGORICAL FEATURES (HANDLE BASELINE)
    # # =========================
    for col_idx, col in enumerate(categorical_features):

        categories = encoder.categories_[col_idx]
        value_counts = X_cat[col].value_counts()

        if encoder.drop_idx_ is not None:
            baseline_idx = encoder.drop_idx_[col_idx]
            baseline = categories[baseline_idx]
        else:
            baseline = None

        rows.append({
            "Feature": f"{col}_{baseline}",
            "Coefficient": 0.0,
            "Count": value_counts.get(baseline, 0)
        })

        # other categories
        for i, val in enumerate(categories):
            if encoder.drop_idx_ is not None and i == encoder.drop_idx_[col_idx]:
                continue
            rows.append({
                "Feature": f"{col}_{str(val)}",
                "Coefficient": next(coef_iter),
                "Count": value_counts.get(val, 0)
            })

    # =========================
    # ADD REMAINING FEATURES (NUMERIC + SPLINE)
    # =========================

    # Get transformed matrix
    X_transformed = preprocessor.transform(X)
    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    for fname, coef in zip(feature_names, coefs):

        clean_name = fname.split("__", 1)[-1]
        if clean_name.startswith(tuple(categorical_features)):
            continue

        # skip spline baseline (already added)
        if spline_drop and any(clean_name == v for v in spline_drop.values()):
            continue

        # IMPORTANT: include ALL features (including missing)
        col_values = X_transformed[clean_name]

        count = sum(col_values)

        rows.append({
            "Feature": clean_name,
            "Coefficient": coef,
            "Count": count
        })

    # =========================
    # ADD SPLINE BASELINES
    # =========================
    if spline_features and spline_drop:

        for var, drop_col in spline_drop.items():
            # find drop_col in feature_names
            center = float(drop_col.rsplit("_", 2)[1])
            breaks = spline_features[var]
            knots = [-np.inf] + breaks + [np.inf]

            idx = knots.index(center)
            feature_name = f"{var}_{_format_break(center)}_{idx:.0f}"

            # Find the number of observations for this variable
             # Total observations
            total_n = len(X)

            # Find all spline columns for this variable (excluding baseline)
            spline_cols = [
                row["Feature"] for row in rows
                if row["Feature"].startswith(f"{var}_") and row["Feature"] != drop_col
            ]

            # Sum counts of all non-baseline spline features
            non_baseline_count = sum(
                row["Count"] for row in rows
                if row["Feature"] in spline_cols
            )

            # Baseline count = total - others
            baseline_count = total_n - non_baseline_count

            rows.append({
                "Feature": feature_name,
                "Coefficient": 0.0,
                "Count": baseline_count
            })

    # NEW: build dataframe from rows
    coef_df = pd.DataFrame(rows)

    # =========================
    # ADD STATISTICS (MERGE SAFE)
    # =========================
    stats_df = compute_sklearn_stats(pipeline, X, y, model_type=model_type)

    # Merge stats into coef_df
    coef_df = coef_df.merge(
        stats_df.drop(columns=["Coefficient"]),
        on="Feature",
        how="left"
    )

    # =========================
    # ADD INTERCEPT + METRICS
    # =========================

    intercept_row = stats_df[stats_df["Feature"] == "Intercept"]
    coef_df = pd.concat([coef_df, intercept_row], ignore_index=True)
    
    if model_type == "linear":
        y_pred = pipeline.predict(X)
        prediction_values = y_pred
        coef_df = pd.concat([
            coef_df,
            pd.DataFrame([{
                "Feature": "R²",
                "Coefficient": pipeline.score(X, y)
            }])
        ], ignore_index=True)

    else:
        # y_pred = pipeline.predict(X)
        # y_prob = pipeline.predict_proba(X)[:, 1]
        # if adjust_logistic_decision_threshold != None:
        #     y_pred = (y_prob > adjust_logistic_decision_threshold)
        
        # print(sum(y), sum(y_pred), len(y))

        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(X)

        # compute logits manually with intercept shift
        logits = (model.intercept_[0] * intercept_multiplier) + X_transformed @ model.coef_.ravel()
        y_prob = 1 / (1 + np.exp(-logits))

        # apply threshold
        threshold = adjust_logistic_decision_threshold if adjust_logistic_decision_threshold is not None else 0.5
        y_pred = (y_prob > threshold).astype(int)
        prediction_values = y_prob

        coef_df = pd.concat([
            coef_df,
            pd.DataFrame([{"Feature": "Accuracy", "Coefficient": accuracy_score(y, y_pred)}]),
            pd.DataFrame([{"Feature": "AUC", "Coefficient": roc_auc_score(y, y_prob)}])
        ], ignore_index=True)

        if plot:
            fpr, tpr, _ = roc_curve(y, y_prob)
            
            plt.close('all')
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()

    coef_df = sort_spline_by_var(coef_df, spline_features)
    front_cols = [
        col for col in ["Feature", "Coefficient", "Count", "P Value"]
        if col in coef_df.columns
    ]
    coef_df = coef_df[
        front_cols + [col for col in coef_df.columns if col not in front_cols]
    ]
    # =========================
    # FEATURE IMPORTANCE
    # =========================
    if plot:
        plot_feature_importance(coef_df, top_n=plot_top_n_features)

    if output_coef:
        coef_df.to_excel(output_file_name, index=False)

    return coef_df, prediction_values

def reduce_categories(df, categorical_features, max_categories):
    """
    For each categorical column:
    - Keep top N most frequent categories
    - Replace others with 'Other'
    
    Returns:
    - transformed df
    - mapping dict (for consistent application to test data)
    """
    df = df.copy()
    category_maps = {}

    for col in categorical_features:
        top_categories = df[col].value_counts().nlargest(max_categories).index
        category_maps[col] = set(top_categories)

        df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")

    return df, category_maps


def apply_category_mapping(df, categorical_features, category_maps):
    df = df.copy()

    for col in categorical_features:
        allowed = category_maps[col]
        df[col] = df[col].apply(lambda x: x if x in allowed else "Other")

    return df


# -------------------------------------------------
# MAIN MODEL PIPELINE   
# -------------------------------------------------
def run_model_pipeline(df,
                       target_col,
                       categorical_features,
                       numeric_features,
                       spline_features=None,
                       manual_baseline=None,
                       spline_manual_baseline=None,
                       model_type="linear",
                       test_size=0,
                       df_test = None,
                       test_target_col=None,
                       max_categories=10,
                       random_state=42,
                       plot=True,
                       output_coef=False,
                       output_file_name="Regression_Coefficients.xlsx",
                       plot_top_n_features = 25,
                       id_col=None,
                       adjust_logistic_decision_threshold=None,
                       intercept_multiplier=1.0,
                       only_plot_corr = False
                       ):

    # =========================
    # PREP DATA
    # =========================
    eval_target_col = test_target_col if test_target_col is not None else target_col
    id_series = df[id_col] if id_col else None
    X = df[categorical_features + numeric_features + (list(spline_features.keys()) if spline_features else [])].copy()
    X = clean_numeric_data(X, numeric_features) 

    y = df[target_col]
    y_eval = df[eval_target_col] if test_size and test_size > 0 else y
    # Drop rows where y is NaN
    mask = y.notna()
    if test_size and test_size > 0:
        mask = mask & y_eval.notna()
    X = X[mask]
    y = y[mask]
    y_eval = y_eval[mask]
    if id_col:
        id_series = id_series[mask]


    # Plot correlation matrix for numeric features
    corr = plot_correlation_simple(X, X.columns, plot=plot)

    if only_plot_corr == True:
        return corr

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    if test_size and test_size > 0:
        if id_col:
            X_train, X_test, y_train, _, _, y_test, id_train, id_test = train_test_split(
                X, y, y_eval, id_series, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, _, _, y_test = train_test_split(
                X, y, y_eval, test_size=test_size, random_state=random_state
            )
            id_train, id_test = None, None
        do_oos = True
    else:
        if df_test is not None:
            X_train, y_train = X, y
            id_train = id_series

            id_test = df_test[id_col] if id_col else None
            X_test = df_test[categorical_features + numeric_features + (list(spline_features.keys()) if spline_features else [])].copy()
            X_test = clean_numeric_data(X_test, numeric_features) 
            y_test = df_test[eval_target_col]

            do_oos = True
        else:
            X_train, y_train = X, y
            id_train = id_series
            X_test, y_test, id_test = None, None, None
            do_oos = False

    # =========================
    # REDUCE CATEGORIES (TRAIN)
    # =========================
    X_train, category_maps = reduce_categories(
        X_train, categorical_features, max_categories=max_categories
    )

    # Apply to test (if exists)
    if do_oos:
        X_test = apply_category_mapping(
            X_test, categorical_features, category_maps
        )

    # =========================
    # TRAIN MODEL
    # =========================
    pipeline, _, _, spline_drop = train_model(
                                            X_train,
                                            y_train,
                                            categorical_features,
                                            numeric_features,
                                            spline_features=spline_features,
                                            manual_baseline=manual_baseline,
                                            spline_manual_baseline=spline_manual_baseline,
                                            model_type=model_type
                                            )

    # =========================
    # IN-SAMPLE SUMMARY
    # =========================
    coef_df, y_pred_train = summarize_model(
                                            pipeline,
                                            X_train,
                                            y_train,
                                            categorical_features,
                                            numeric_features,
                                            spline_features=spline_features,
                                            spline_drop=spline_drop,
                                            model_type=model_type,
                                            output_coef=output_coef,
                                            output_file_name=output_file_name,
                                            plot=plot,
                                            plot_top_n_features=plot_top_n_features,
                                            adjust_logistic_decision_threshold=adjust_logistic_decision_threshold,
                                            intercept_multiplier = intercept_multiplier
                                        )
    
    X_train_with_preds = X_train.copy()
    if id_col:
        X_train_with_preds[id_col] = id_train
    X_train_with_preds["y_true"] = y_train
    X_train_with_preds["y_pred"] = y_pred_train

    results = {}

    # =========================
    # OUT-OF-SAMPLE (OPTIONAL)
    # =========================
    if do_oos:

        # y_pred_test = pipeline.predict(X_test)

        if model_type == "logistic":
            preprocessor = pipeline.named_steps["preprocess"]
            model = pipeline.named_steps["model"]

            X_test_transformed = preprocessor.transform(X_test)

            logits_test = (model.intercept_[0] * intercept_multiplier) + X_test_transformed @ model.coef_.ravel()
            y_prob_test = 1 / (1 + np.exp(-logits_test))

            threshold = adjust_logistic_decision_threshold if adjust_logistic_decision_threshold is not None else 0.5
            y_pred_test = (y_prob_test > threshold).astype(int)
            test_pred_values = y_prob_test
        else:
            y_pred_test = pipeline.predict(X_test)
            test_pred_values = y_pred_test

        X_test_with_preds = X_test.copy()
        if id_col:
            X_test_with_preds[id_col] = id_test
        X_test_with_preds["y_true"] = y_test
        X_test_with_preds["y_pred"] = test_pred_values

        if model_type == "linear":

            results["RMSE_train"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
            results["RMSE_test"] = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results["R2_train"] = r2_score(y_train, y_pred_train)
            results["R2_test"] = r2_score(y_test, y_pred_test)

        else:

            y_prob_train = y_pred_train
            y_prob_test = test_pred_values

            results["AUC_train"] = roc_auc_score(y_train, y_prob_train)
            results["AUC_test"] = roc_auc_score(y_test, y_prob_test)

            # ROC plot (train vs test)
            if plot:

                fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
                fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

                plt.close('all')
                plt.figure()
                plt.plot(fpr_train, tpr_train, label=f"Train AUC = {results['AUC_train']:.3f}")
                plt.plot(fpr_test, tpr_test, label=f"Test AUC = {results['AUC_test']:.3f}")
                plt.plot([0, 1], [0, 1], linestyle="--")

                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve (Train vs Test)")
                plt.legend()
                plt.tight_layout()
                plt.show()

    # =========================
    # RETURN
    # =========================
    return {
        "pipeline": pipeline,
        "coef_df": coef_df,
        "X_train_with_preds": X_train_with_preds,
        "outofsample_results": results,
        "X_test_with_preds": X_test_with_preds if do_oos else None,
        "has_oos": do_oos
    }


def run_residual_layer_pipeline(
    df,
    target_col,
    base_categorical_features,
    base_numeric_features,
    residual_categorical_features,
    residual_numeric_features,
    base_spline_features=None,
    residual_spline_features=None,
    base_manual_baseline=None,
    residual_manual_baseline=None,
    base_spline_manual_baseline=None,
    residual_spline_manual_baseline=None,
    base_model_type="logistic",
    residual_model_type="linear",
    test_size=0,
    df_test=None,
    test_target_col=None,
    max_categories=10,
    residual_max_categories=None,
    random_state=42,
    plot=True,
    output_coef=False,
    base_output_file_name="Stage1_Regression_Coefficients.xlsx",
    residual_output_file_name="Stage2_Residual_Regression_Coefficients.xlsx",
    plot_top_n_features=25,
    id_col=None,
    adjust_logistic_decision_threshold=None,
    intercept_multiplier=1.0,
    residual_col="stage1_residual",
    stage1_pred_col="stage1_pred",
    stage2_pred_col="stage2_residual_pred",
    combined_pred_col="combined_pred",
    clip_combined_pred=True
):
    """
    Run a two-layer model:
    1. Fit the base model on fundamental factors.
    2. Fit a second model on the base-model residual.
    """
    residual_max_categories = (
        residual_max_categories
        if residual_max_categories is not None
        else max_categories
    )

    stage1_results = run_model_pipeline(
        df=df,
        target_col=target_col,
        categorical_features=base_categorical_features,
        numeric_features=base_numeric_features,
        spline_features=base_spline_features,
        manual_baseline=base_manual_baseline,
        spline_manual_baseline=base_spline_manual_baseline,
        model_type=base_model_type,
        test_size=test_size,
        df_test=df_test,
        test_target_col=test_target_col,
        max_categories=max_categories,
        random_state=random_state,
        plot=plot,
        output_coef=output_coef,
        output_file_name=base_output_file_name,
        plot_top_n_features=plot_top_n_features,
        id_col=id_col,
        adjust_logistic_decision_threshold=adjust_logistic_decision_threshold,
        intercept_multiplier=intercept_multiplier
    )

    def build_residual_df(source_df, preds_df):
        out = source_df.loc[preds_df.index].copy()
        out[stage1_pred_col] = preds_df["y_pred"].to_numpy()
        out[residual_col] = (
            preds_df["y_true"].to_numpy()
            - preds_df["y_pred"].to_numpy()
        )
        return out

    stage2_train_df = build_residual_df(
        df,
        stage1_results["X_train_with_preds"]
    )

    stage2_test_df = None
    if stage1_results["has_oos"]:
        test_source_df = df if test_size and test_size > 0 else df_test
        stage2_test_df = build_residual_df(
            test_source_df,
            stage1_results["X_test_with_preds"]
        )

    stage2_results = run_model_pipeline(
        df=stage2_train_df,
        target_col=residual_col,
        categorical_features=residual_categorical_features,
        numeric_features=residual_numeric_features,
        spline_features=residual_spline_features,
        manual_baseline=residual_manual_baseline,
        spline_manual_baseline=residual_spline_manual_baseline,
        model_type=residual_model_type,
        test_size=0,
        df_test=stage2_test_df,
        max_categories=residual_max_categories,
        random_state=random_state,
        plot=plot,
        output_coef=output_coef,
        output_file_name=residual_output_file_name,
        plot_top_n_features=plot_top_n_features,
        id_col=id_col
    )

    def build_combined_preds(stage1_preds_df, stage2_preds_df):
        combined = stage2_preds_df.copy().rename(
            columns={
                "y_true": residual_col,
                "y_pred": stage2_pred_col
            }
        )
        combined["y_true"] = stage1_preds_df["y_true"].to_numpy()
        combined[stage1_pred_col] = stage1_preds_df["y_pred"].to_numpy()
        combined[combined_pred_col] = (
            combined[stage1_pred_col] + combined[stage2_pred_col]
        )

        if clip_combined_pred and base_model_type == "logistic":
            combined[combined_pred_col] = combined[combined_pred_col].clip(0, 1)

        combined["y_pred"] = combined[combined_pred_col]
        return combined

    combined_train_with_preds = build_combined_preds(
        stage1_results["X_train_with_preds"],
        stage2_results["X_train_with_preds"]
    )

    combined_test_with_preds = None
    if stage1_results["has_oos"]:
        combined_test_with_preds = build_combined_preds(
            stage1_results["X_test_with_preds"],
            stage2_results["X_test_with_preds"]
        )

    def score_combined(preds_df, prefix):
        if preds_df is None:
            return {}

        y_true = preds_df["y_true"]
        y_pred = preds_df[combined_pred_col]

        if base_model_type == "linear":
            return {
                f"RMSE_{prefix}": np.sqrt(mean_squared_error(y_true, y_pred)),
                f"R2_{prefix}": r2_score(y_true, y_pred)
            }

        threshold = (
            adjust_logistic_decision_threshold
            if adjust_logistic_decision_threshold is not None
            else 0.5
        )
        metrics = {
            f"Accuracy_{prefix}": accuracy_score(
                y_true,
                (y_pred > threshold).astype(int)
            )
        }

        try:
            metrics[f"AUC_{prefix}"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            metrics[f"AUC_{prefix}"] = np.nan

        return metrics

    combined_results = {}
    combined_results.update(score_combined(combined_train_with_preds, "train"))
    combined_results.update(score_combined(combined_test_with_preds, "test"))

    return {
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "combined_train_with_preds": combined_train_with_preds,
        "combined_test_with_preds": combined_test_with_preds,
        "combined_results": combined_results,
        "residual_col": residual_col,
        "stage1_pred_col": stage1_pred_col,
        "stage2_pred_col": stage2_pred_col,
        "combined_pred_col": combined_pred_col
    }


def auc_drop_test(df_training, 
                  target_col, 
                  categorical_features, 
                  numeric_features, 
                  spline_features, 
                  manual_baseline,
                  model_type="logistic",
                  test_size=0,
                  max_categories=10,
                  random_state=42):
    
    features = categorical_features + numeric_features + list(spline_features.keys())
    output_list = []
    for i in range(len(features)+1):
        if i == 0:
            results = run_model_pipeline(
                        df = df_training,
                        target_col=target_col,
                        categorical_features=categorical_features,
                        numeric_features=numeric_features,
                        spline_features=spline_features, 
                        manual_baseline = manual_baseline,
                        model_type=model_type,
                        
                        test_size=test_size,
                        max_categories=max_categories,
                        plot_top_n_features = None,
                        random_state=random_state,
                        
                        plot=False,
                        output_coef=False,
                        # output_file_name="Outputs/Regression_Coefficients.xlsx"
                    )
            
            base_auc = results['coef_df'].loc[len(results['coef_df']) - 1, 'Coefficient']
            output_list.append(base_auc)

        else:
            drop_col = features[i-1]

            if drop_col not in df_training.columns:
                print(f"Warning: {drop_col} not found in training data columns.")
            else:
                df_training_ = df_training.drop(columns=drop_col).copy()

            if drop_col in spline_features.keys():
                spline_features_copy = spline_features.copy()
                spline_features_copy.pop(drop_col)
            else:
                spline_features_copy = spline_features.copy()

            
            if drop_col in categorical_features:
                categorical_features_input = categorical_features.copy()
                categorical_features_input.remove(drop_col)
            else:
                categorical_features_input = categorical_features.copy()
            
            results = run_model_pipeline(
                df = df_training_,
                target_col=target_col,
                categorical_features=categorical_features_input,
                numeric_features=numeric_features,
                spline_features=spline_features_copy,
                manual_baseline = manual_baseline,
                model_type=model_type,
                
                test_size=test_size,
                max_categories=max_categories,
                plot_top_n_features = None,
                random_state=random_state,
                plot=False,
                output_coef=False,
                # output_file_name="Outputs/Regression_Coefficients.xlsx"
            )
                
            auc = results['coef_df'].loc[len(results['coef_df']) - 1, 'Coefficient']
            output_list.append(auc)

    res_df = pd.DataFrame([['All'] + features, output_list]).T
    res_df.columns = ['Variables', 'AUC w/o Variable']
    res_df['Base AUC'] = base_auc
    res_df['AUC Drop'] = res_df['AUC w/o Variable'] - base_auc
    res_df = res_df[res_df['Variables'] != 'All']
    res_df.sort_values('AUC Drop', ascending=True, inplace=True)
    res_df['Ranking'] = res_df['AUC Drop'].rank(method='min', ascending=True).astype(int)
    res_df.reset_index(drop=True, inplace=True)

    return res_df


def get_table_for_plot_in_sample_vs_oos(results, 
                                        df_training, 
                                        df_test, 
                                        df_origination, 
                                        secondary_y_cols, 
                                        orig_bal_col_name,
                                        id_col_name,
                                        vintage_col_name,
                                        categorical_col_to_plot = None,
                                        empirical_col = None
                                        ):
                    
    if empirical_col != None:
        in_sample_outputs = df_training[[id_col_name, empirical_col]]
        in_sample_outputs.rename(columns = {empirical_col: 'y_true'},  inplace=True)
    else:
        in_sample_outputs = results['X_train_with_preds']

    oos_result = results['X_test_with_preds']
    # oos_result = oos_result[oos_result['Credit Rating Bucket'] == 'B']

    if orig_bal_col_name not in in_sample_outputs.columns:
        in_sample_outputs = in_sample_outputs.merge(df_training[[id_col_name, vintage_col_name, orig_bal_col_name]], on = id_col_name, how = 'left', suffixes=('', '_added'))
    else:
        in_sample_outputs = in_sample_outputs.merge(df_training[[id_col_name, vintage_col_name]], on = id_col_name, how = 'left', suffixes=('', '_added'))
    in_sample_outputs['Empirical Amount'] = in_sample_outputs.loc[:, 'y_true'] * in_sample_outputs.loc[:, orig_bal_col_name]
    # print(in_sample_outputs.head())

    if orig_bal_col_name not in oos_result.columns:
        oos_result = oos_result.merge(df_test[[id_col_name, vintage_col_name, orig_bal_col_name]], on = id_col_name, how = 'left', suffixes=('', '_added'))
    else:
        oos_result = oos_result.merge(df_test[[id_col_name, vintage_col_name]], on = id_col_name, how = 'left', suffixes=('', '_added'))
    oos_result['Model Results Amount'] = oos_result.loc[:, 'y_pred'] * oos_result.loc[:, orig_bal_col_name]
    print(oos_result.columns)
    output_oos = oos_result.groupby([vintage_col_name]).sum()[[orig_bal_col_name, 'Model Results Amount']]
    output_oos['Model Results'] = output_oos['Model Results Amount'] / output_oos[orig_bal_col_name]

    output = in_sample_outputs.groupby([vintage_col_name]).sum()[[orig_bal_col_name, 'Empirical Amount']]
    output['Empirical'] = output['Empirical Amount'] / output[orig_bal_col_name]
    # output = output.apply(trim_last_n, n=trim_n )

    output = output.join(output_oos, how='outer', lsuffix='_in_sample', rsuffix='_oos')

    if categorical_col_to_plot:
        percentage_tables = get_wt_distribution(df_origination, 
                                                    secondary_y_cols[0],
                                                    vintage = vintage_col_name,
                                                    weight_by = orig_bal_col_name, show_total_origination = False, format_pct = False).T
        output = output.join(percentage_tables[[categorical_col_to_plot]], how='outer')        
        output_plot = output[['Empirical', 'Model Results'] + [categorical_col_to_plot]]  
        # secondary_ylabel = '{} = {} %'.format(secondary_y_cols[0], 'Yes' if categorical_col_to_plot == 1 else categorical_col_to_plot)
        secondary_y_cols = [categorical_col_to_plot]   
        # secondary_percentage = True                               
    else:
        wa_tables = groupby_weighted_avg(df_origination, groupby_col = vintage_col_name, cols = secondary_y_cols, weight_col = orig_bal_col_name)
        output = output.join(wa_tables, how='outer')    

        output_plot = output[['Empirical', 'Model Results'] + secondary_y_cols]
        # secondary_percentage = False

    # if secondary_y_cols:
    #     plot_finance_style(
    #                         # everdp_plot,
    #                         output_plot, 
    #                         title="Ever D{:.0f} at MOB {:.0f} vs {}".format(ever_n_days_past_due, loan_age_cutoff, secondary_ylabel ),
    #                         ylabel="Ever D{:.0f} at MOB {:.0f}".format(ever_n_days_past_due, loan_age_cutoff), data_labels=False,
    #                         secondary_y_cols=secondary_y_cols,   # columns on right axis
    #                         secondary_ylabel = secondary_ylabel,
    #                         percentage=True,
    #                         secondary_percentage=secondary_percentage,
    #                         use_custom_colors=False,
    #                         linestyles_inputs = ['-', '-', '--']
    #                         )
    # else:
    #     plot_finance_style(
    #                         # everdp_plot,
    #                         output_plot, 
    #                         title="Ever D{:.0f} at MOB {:.0f}".format(ever_n_days_past_due, loan_age_cutoff),
    #                         ylabel="Ever D{:.0f} at MOB {:.0f}".format(ever_n_days_past_due, loan_age_cutoff), data_labels=False,
    #                         # secondary_y_cols=secondary_y_cols,   # columns on right axis
    #                         # secondary_ylabel = secondary_ylabel,
    #                         percentage=True,
    #                         secondary_percentage=secondary_percentage,
    #                         use_custom_colors=False,
    #                         linestyles_inputs = ['-', '-', '--']
    #                         )

    return output_plot
