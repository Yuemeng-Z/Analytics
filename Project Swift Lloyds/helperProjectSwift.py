import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from parentHelpers import plot_finance_style
import matplotlib.ticker as mtick
import matplotlib.dates as mdates


EMPLOYMENT_STATUS_BUCKETS = {
    "EMRS": "EMRS - Employed, private sector",
    "EMBL": "EMBL - Employed, public sector",
    "EMUK": "EMUK - Employed, sector unknown",
    "UNEM": "UNEM - Unemployed",
    "SFEM": "SFEM - Self-employed",
    "SFE": "SFE - Self-employed",
    "NOEM": "NOEM - No employment, legal entity obligor",
    "STNT": "STNT - Student",
    "PNNR": "PNNR - Pensioner",
    "OTHR": "OTHR - Other",

    "ND":  "OTHR - Other",
    "9": "PNNR - Pensioner",
    "10": "OTHR - Other",
    "6": "SFEM - Self-employed",
    "1": "EMUK - Employed, sector unknown"
}

ORIGINATION_CHANNEL_BUCKETS = {
    "WEBI": "WEBI - Internet",
    "BRCH": "BRCH - Branch",
    "TLSL": "TLSL - Telesale",
    "STND": "STND - Stand",
    "POST": "POST - Post",
    "WLBL": "WLBL - White Label",
    "MGZN": "MGZN - Magazine",
    "ADLR": "ADLR - Automobile Dealer",
    "OTHR": "OTHR - Other",
}

PURPOSE_BUCKETS = {
    "TUIT": "TUIT - Tuition",
    "LEXP": "LEXP - Living Expenses",
    "MDCL": "MDCL - Medical",
    "HIMP": "HIMP - Home Improvement",
    "APFR": "APFR - Appliance or Furniture",
    "TRVL": "TRVL - Travel",
    "DCON": "DCON - Debt Consolidation",
    "NCAR": "NCAR - New Car",
    "UCAR": "UCAR - Used Car",
    "OTHV": "OTHV - Other Vehicle",
    "EQUP": "EQUP - Equipment",
    "PROP": "PROP - Property",
    "OTHR": "OTHR - Other",
    '13': "OTHR - Other",
    '7': "DCON - Debt Consolidation",
    '10': "OTHV - Other Vehicle",
    '6': "TRVL - Travel",
    '4': "HIMP - Home Improvement"
}

RESIDENT_BUCKETS = {
    "1": "1 - Resident less than 3 years",
    "2": "2 - Resident >= 3 years",
    "3": "3 - Not Resident",
    "ND": "ND - No Data",
}


def _map_code_bucket(series, mapping, unknown_label):
    code = series.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    unknown_code = code.where(code.isna(), code + f" - {unknown_label}")

    return code.map(mapping).fillna(unknown_code)


def _format_region_label(value):
    if pd.isna(value):
        return value

    text = str(value).strip()
    abbreviations = {"NE", "NW", "SE", "SW", "UK", "GB", "NI"}

    return " ".join(
        part.upper() if part.upper() in abbreviations else part.lower().title()
        for part in text.split("_")
    )


def add_employment_status_bucket(df):
    if ("Employment_Status" not in df.columns) and ("Borrower_Employment_Status" not in df.columns):
        return df.copy()

    result = df.copy()

    if "Borrower_Employment_Status" in result.columns:
        result["Employment_Status_Bucket"] = _map_code_bucket(
            result["Borrower_Employment_Status"],
            EMPLOYMENT_STATUS_BUCKETS,
            "Unknown employment status code",
        )
    else:
        result["Employment_Status_Bucket"] = _map_code_bucket(
            result["Employment_Status"],
            EMPLOYMENT_STATUS_BUCKETS,
            "Unknown employment status code",
        )

    if "Borrower_Employment_Status" in result.columns:
            result["Employment_Status_Bucket"] = _map_code_bucket(
            result["Borrower_Employment_Status"],
            EMPLOYMENT_STATUS_BUCKETS,
            "Unknown employment status code",
        )

    return result


def add_readable_category_buckets(df):
    result = add_employment_status_bucket(df)

    if "Origination_Channel" in result.columns:
        result["Origination_Channel_Bucket"] = _map_code_bucket(
            result["Origination_Channel"],
            ORIGINATION_CHANNEL_BUCKETS,
            "Unknown origination channel code",
        )

    if "Purpose" in result.columns:
        result["Purpose_Bucket"] = _map_code_bucket(
            result["Purpose"],
            PURPOSE_BUCKETS,
            "Unknown purpose code",
        )

    if "LOANS_PURPOSE" in result.columns:
        result["Loan_Purpose_Mapped_Bucket"] = _map_code_bucket(
            result["LOANS_PURPOSE"],
            PURPOSE_BUCKETS,
            "Unknown purpose code",
        )

    if "Loan_Purpose_Mapped" in result.columns:
        result["Loan_Purpose_Mapped_Bucket"] = _map_code_bucket(
            result["Loan_Purpose_Mapped"],
            PURPOSE_BUCKETS,
            "Unknown purpose code",
        )

    if "Resident" in result.columns:
        result["Resident_Bucket"] = _map_code_bucket(
            result["Resident"],
            RESIDENT_BUCKETS,
            "Unknown resident code",
        )

    if "Post Code Region Application" in result.columns:
        result["Post Code Region Application_Bucket"] = (
            result["Post Code Region Application"].map(_format_region_label)
        )

    if "Geographic_Region" in result.columns:
        result["Post Code Region Application_Bucket"] = (
            result["Geographic_Region"].map(_format_region_label)
        )

    return result


def clean_loan_tape(LoanTape):

    print('Duplicated Loan Count: ',
      LoanTape.shape[0] - LoanTape['New_Underlying_Exposure_Identifier'].nunique()
     )

    LoanTape = LoanTape.drop_duplicates(
        subset=['New_Underlying_Exposure_Identifier'],
        keep='first'
    )

    print('Final Loan Tape Shape: ', LoanTape.shape)

    LoanTape = LoanTape[['EverD60CO_Mar2026',
                     'EverD60CO_Sep2025',
                    'New_Underlying_Exposure_Identifier',
                    'Employment_Status',
                    'Credit_Impaired_Obligor',
                    'Primary_Income',
                    'Origination_Channel',
                    'Purpose',
                    'Origination_Date',
                    'Original_Term',
                    'Original_Principal_Balance',
                    'Current_Interest_Rate',
                    'Borrower Credit Quality',
                    'Number of Borrowers',
                    'Resident',
                    'Bureau Score Value',
                    'STAGE',
                    'AGE',
                    'Post Code Region Application',
                    'Set off amount',
                    'Loan_Purpose_Mapped',
                    'LoanAgeDec2023',
                    'Dec2023Bal'
                ]]

    LoanTape = add_readable_category_buckets(LoanTape)
    # print('Added readable category buckets.', LoanTape.columns)
    
    LoanTape['Origination_Date'] = pd.to_datetime(LoanTape['Origination_Date'], errors='coerce')

    LoanTape['Origination_Quarter'] = (
        LoanTape['Origination_Date'].dt.to_period('Q').astype(str)
    )


    quarter_avg_rate = (
        LoanTape
        .groupby('Origination_Quarter')
        .apply(lambda g: np.average(g['Current_Interest_Rate'], 
                                    weights=g['Dec2023Bal']))
        .rename('weighted_quarter_rate')
        .reset_index()
    )

    # Merge back to original LoanTape
    LoanTape = LoanTape.merge(quarter_avg_rate, on='Origination_Quarter', how='left')

    LoanTape['normalized_interest_rate'] = (
        LoanTape['Current_Interest_Rate'] - LoanTape['weighted_quarter_rate']
    )


    r = LoanTape['Current_Interest_Rate'] / 100 / 12
    n = LoanTape['Original_Term']
    bal = LoanTape['Original_Principal_Balance']

    # Monthly payment formula, with zero-rate handling
    LoanTape['MonthlyPayment'] = np.where(
        r == 0,
        bal / n,                                       # zero-interest case
        r * bal / (1 - (1 + r)**(-n))                  # standard amortization
    )

    LoanTape['PTI'] = LoanTape['MonthlyPayment'] / LoanTape['Primary_Income'] * 12 * 100

    LoanTape['Self_Employed_or_Not'] = LoanTape['Employment_Status'].apply(lambda x: 'Yes' if x == 'SFEM' else 'No')
    LoanTape['Loan_Purpose'] = LoanTape['Purpose_Bucket'].apply(lambda x: x if x in ['OTHV - Other Vehicle', 'HIMP - Home Improvement'] 
                                                                else 'Debt Consolidation, Travel or Other')
    LoanTape['Region'] = LoanTape['Post Code Region Application_Bucket'].apply(lambda x: x if x in ['London'] 
                                                                                else 'Outside London')
    
    LoanTape['Has 2 Borrowers'] = np.where(LoanTape['Number of Borrowers'] == 2, 1, 0)
    LoanTape['Has 2 Borrowers x Borrower Credit Quality'] = LoanTape['Has 2 Borrowers'] * LoanTape['Borrower Credit Quality']

    LoanTape['Has 1 Borrowers'] = np.where(LoanTape['Number of Borrowers'] == 1, 1, 0)
    LoanTape['Has 1 Borrowers x Borrower Credit Quality'] = LoanTape['Has 1 Borrowers'] * LoanTape['Borrower Credit Quality']
        
    return LoanTape


def plot_everD60_by_features(df, everD60_col='EverD60CO_Mar2026'):
    df = add_readable_category_buckets(df)
    
    cols = [
        everD60_col,
        "Employment_Status_Bucket",
        'Self_Employed_or_Not',

        "Credit_Impaired_Obligor",
        "Primary_Income",
        "Origination_Channel_Bucket",
        "Purpose_Bucket",
        "Loan_Purpose",
        "Origination_Date",
        "Original_Term",
        "Original_Principal_Balance",
        "normalized_interest_rate",
        "Borrower Credit Quality",
        "Number of Borrowers",
        "Resident_Bucket",
        "Bureau Score Value",
        'Set off amount',
        "STAGE",
        "AGE",
        "Post Code Region Application_Bucket",
        "Region",
        # "Loan_Purpose_Mapped_Bucket",
        "LoanAgeDec2023",
        "Dec2023Bal",
        "Original_Principal_Balance",
        'PTI',
        'MonthlyPayment',
        'Primary_Income',
        'LoanAgeDec2023',
        # "Has 2 Borrowers x Borrower Credit Quality"
    ]

    for col in cols:
        if col not in df.columns or col in [everD60_col]:
            continue
        plot_one_visualization(df, col, everD60_col=everD60_col)
        

def plot_one_visualization(df, col, everD60_col='EverD60CO_Mar2026', breakpoints=None, n_bins=10):
    # if col not in df.columns or col in ["EverD60CO", "Dec2023Bal"]:

    display_col = col.replace("_Bucket", "").replace("_", " ")
    s = df[col]
    n_unique = s.nunique(dropna=True)
    is_categorical = (s.dtype == "object") or (s.dtype.name == "category")

    # ---------- CASE 1: categorical or <= 10 unique values ----------
    if breakpoints is None and (is_categorical or n_unique <= 10):
        grouped = df.groupby(col, dropna=False, sort=False)
        pct = grouped.apply(
            lambda g: g.loc[g[everD60_col] == 1, "Dec2023Bal"].sum() / g["Dec2023Bal"].sum()
            if g["Dec2023Bal"].sum() != 0 else 0
        )
        counts = grouped.size()

        plt.figure(figsize=(11, 5))
        ax = pct.plot(kind="bar", color="#4C78A8", edgecolor="white", width=0.8)
        ymax = pct.max() if len(pct) else 0
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel("DQ60/CO% of orig bal\n(defined as Dec-2023 balance) as of 3/31/2026")
        ax.set_xlabel("")
        ax.set_title(f"DQ60/CO Rate by {display_col}", fontsize=13, weight="bold")
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
        pct = grouped.apply(
            lambda g: g.loc[g[everD60_col] == 1, "Dec2023Bal"].sum() / g["Dec2023Bal"].sum()
            if g["Dec2023Bal"].sum() != 0 else 0
        )
        counts = grouped.size()

        plt.figure(figsize=(11, 5))
        ax = pct.plot(kind="bar", color="#4C78A8", edgecolor="white", width=0.8)
        ymax = pct.max() if len(pct) else 0
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel("DQ60/CO% of orig bal\n(defined as Dec-2023 balance) as of 3/31/2026")
        ax.set_xlabel("")
        ax.set_title(f"DQ60/CO Rate by {display_col}", fontsize=13, weight="bold")
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


def make_balance_deciles(df, score_col, bal_col, bad_col, n_deciles=10):
        df_sorted = df.sort_values(score_col, ascending=True).copy()
        
        df_sorted['cum_bal'] = df_sorted[bal_col].cumsum()
        total_bal = df_sorted[bal_col].sum()
        df_sorted['bal_pct'] = df_sorted['cum_bal'] / total_bal
        
        df_sorted['decile'] = np.ceil(df_sorted['bal_pct'] * n_deciles).astype(int)
        df_sorted.loc[df_sorted['decile'] < 1, 'decile'] = 1
        df_sorted.loc[df_sorted['decile'] > n_deciles, 'decile'] = n_deciles
        
        agg = (
            df_sorted
            .groupby('decile')
            .apply(lambda d: pd.Series({
                'Total_Bal': d[bal_col].sum(),
                'Bad_Bal': d.loc[d[bad_col] == 1, bal_col].sum()
            }))
            .reset_index()
        )
        agg['Pct_Bad_Bal'] = agg['Bad_Bal'] / agg['Total_Bal'] * 100
        agg = agg.sort_values('decile')
        
        return agg, df_sorted


def return_combined_results_for_plotting(residual_results, df,
                                         result_data = 'combined_train_with_preds',
                                         bad_col='EverD60CO_Sep2025',
                                         bal_col='Dec2023Bal',
                                         lloyds_score_col='Borrower Credit Quality',
                                         n_deciles=10):
    
    train_result_stage_1 = residual_results[result_data][['New_Underlying_Exposure_Identifier', residual_results['stage1_pred_col']]]
    train_result_stage_2 = residual_results[result_data][['New_Underlying_Exposure_Identifier', residual_results['combined_pred_col']]]

    train_result_stage_1 = add_additional_columns_for_plotting_prediction_results(df, train_result_stage_1, bal_col=bal_col)
    train_result_stage_2 = add_additional_columns_for_plotting_prediction_results(df, train_result_stage_2, bal_col=bal_col)

    agg_model_train_stage_1_sep2025, agg_bcq_train_stage_1_sep2025, df_sorted_model_train_stage_1_sep2025, df_sorted_bcq_train_stage_1_sep2025 = get_decile_boundaries(train_result_stage_1, 
                            bal_col=bal_col, 
                            bad_col=bad_col,
                            score_col=residual_results['stage1_pred_col'],  # this is the BV score model column
                            lloyds_score_col=lloyds_score_col,  # this is the Lloyds score column
                            n_deciles=n_deciles)

    agg_model_train_stage_2_sep2025, agg_bcq_train_stage_2_sep2025, df_sorted_model_train_stage_2_sep2025, df_sorted_bcq_train_stage_2_sep2025 = get_decile_boundaries(train_result_stage_2, 
                        bal_col=bal_col, 
                        bad_col=bad_col,
                        score_col=residual_results['combined_pred_col'],  # this is the BV score model column
                        lloyds_score_col=lloyds_score_col,  # this is the Lloyds score column
                        n_deciles=n_deciles)
    

    train_stage_1_score_dq60_pct = agg_model_train_stage_1_sep2025[['decile', 'Pct_Bad_Bal']]
    train_stage_1_score_dq60_pct.rename(columns={'Pct_Bad_Bal': 'Stage 1 Model'}, inplace=True)
    train_lloyds_score_dq60_pct = agg_bcq_train_stage_1_sep2025[['Pct_Bad_Bal']]
    train_lloyds_score_dq60_pct.rename(columns={'Pct_Bad_Bal': 'Lloyds Score: Borrower Credit Quality'}, inplace=True)
    train_stage_2_score_dq60_pct = agg_model_train_stage_2_sep2025[['Pct_Bad_Bal']]
    train_stage_2_score_dq60_pct.rename(columns={'Pct_Bad_Bal': 'Stage 2: Stage 1 + Interest Rate Residual Model'}, inplace=True)
    res_train_plot = train_lloyds_score_dq60_pct.join(train_stage_1_score_dq60_pct).join(train_stage_2_score_dq60_pct)

    return res_train_plot

def plot_stage_1_2_compare(plot_df, title):
    plt.figure(figsize=(10, 6))

    plt.plot(
        plot_df['decile'].to_numpy(),
        plot_df['Stage 1 Model'].to_numpy(),
        marker='o',
        label='Stage 1 Model Score'
    )

    plt.plot(
        plot_df['decile'].to_numpy(),
        plot_df['Stage 2: Stage 1 + Interest Rate Residual Model'].to_numpy(),
        marker='o',
        linestyle='-',
        label='Stage 2: Stage 1 + Interest Rate Residual Model'
    )

    plt.plot(
        plot_df['decile'].to_numpy(),
        plot_df['Lloyds Score: Borrower Credit Quality'].to_numpy(),
        marker='o',
        linestyle='--',
        label='Lloyds Score: Borrower Credit Quality'
    )

    plt.xlabel('Decile (1 = worst risk, 10 = best risk)')
    plt.ylabel(f'% of Dec2023 Bal that is EverD60CO')
    plt.title(title)
    plt.grid(True)
    plt.xticks(range(1, len(plot_df['decile'].to_numpy()) + 1))
    plt.legend()
    plt.show()

def get_decile_boundaries(test_result, 
                        bal_col='Dec2023Bal', 
                        bad_col='y_true',
                        score_col='y_pred',  # this is the BV score model column
                        lloyds_score_col='Borrower Credit Quality',  # this is the Lloyds score column
                        n_deciles=10,):
    test_result['Score_model'] = -test_result[score_col]

    cols_needed = ['New_Underlying_Exposure_Identifier', 'Score_model', lloyds_score_col, bal_col, bad_col]
    df_all = test_result[cols_needed].dropna()

    df_all['Dec2023Bal'] = pd.to_numeric(df_all['Dec2023Bal'], errors='coerce').fillna(0)
    df_all.loc[df_all['Dec2023Bal'] < 0, 'Dec2023Bal'] = 0

    agg_model, df_sorted_model = make_balance_deciles(df_all, 'Score_model', bal_col, bad_col, n_deciles=n_deciles)
    agg_bcq, df_sorted_bcq   = make_balance_deciles(df_all, lloyds_score_col, bal_col, bad_col, n_deciles=n_deciles)

    return agg_model, agg_bcq, df_sorted_model, df_sorted_bcq


def plot_prediction_results(test_result, 
                            bal_col='Dec2023Bal', 
                            bad_col='y_true',
                            score_col='y_pred',  # this is the BV score model column
                            lloyds_score_col='Borrower Credit Quality',  # this is the Lloyds score column
                            n_deciles=10,
                            title = 'Balance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality (Test Set)'):

    agg_model, agg_bcq, df_sorted_model, df_sorted_bcq = get_decile_boundaries(test_result, 
                                                bal_col=bal_col, 
                                                bad_col=bad_col,
                                                score_col=score_col,  # this is the BV score model column
                                                lloyds_score_col=lloyds_score_col,  # this is the Lloyds score column
                                                n_deciles=n_deciles,)

    # print("Model deciles (balance-weighted):")
    # agg_model.to_clipboard(index=False)

    # print(f"\n{lloyds_score_col} deciles (balance-weighted):")
    # agg_bcq.to_clipboard()

    plt.figure(figsize=(8, 5))

    plt.plot(
        agg_model['decile'].to_numpy(),
        agg_model['Pct_Bad_Bal'].to_numpy(),
        marker='o',
        label='Model Score'
    )

    plt.plot(
        agg_bcq['decile'].to_numpy(),
        agg_bcq['Pct_Bad_Bal'].to_numpy(),
        marker='o',
        linestyle='--',
        label=lloyds_score_col
    )

    plt.xlabel('Decile (1 = worst risk, 10 = best risk)')
    plt.ylabel(f'% of {bal_col} that is EverD60CO')
    plt.title(title)
    plt.grid(True)
    plt.xticks(range(1, n_deciles + 1))
    plt.legend()
    plt.show()

    return agg_model, agg_bcq

def add_additional_columns_for_plotting_prediction_results(df, train_result, bal_col='Dec2023Bal'):
        # if 'Borrower Credit Quality' not in test_result.columns:
        #     test_result = test_result.merge(df[['New_Underlying_Exposure_Identifier', 'Borrower Credit Quality']], on='New_Underlying_Exposure_Identifier', how='left')
        if 'Borrower Credit Quality' not in train_result.columns:
            train_result = train_result.merge(df[['New_Underlying_Exposure_Identifier', 'Borrower Credit Quality']], on='New_Underlying_Exposure_Identifier', how='left')


        # if bal_col not in test_result.columns:
        #     test_result = test_result.merge(df[['New_Underlying_Exposure_Identifier', bal_col]], on='New_Underlying_Exposure_Identifier', how='left')
        if bal_col not in train_result.columns:
            train_result = train_result.merge(df[['New_Underlying_Exposure_Identifier', bal_col]], on='New_Underlying_Exposure_Identifier', how='left')

        # test_result = test_result.merge(df[['New_Underlying_Exposure_Identifier', 'EverD60CO_Sep2025', 'EverD60CO_Mar2026']], on='New_Underlying_Exposure_Identifier', how='left')
        train_result = train_result.merge(df[['New_Underlying_Exposure_Identifier', 'EverD60CO_Sep2025', 'EverD60CO_Mar2026']], on='New_Underlying_Exposure_Identifier', how='left')

        # test_result['new_default_flag'] = np.where(test_result['EverD60CO_Sep2025'] == 1, 0, test_result['EverD60CO_Mar2026'])
        train_result['new_default_flag'] = np.where(train_result['EverD60CO_Sep2025'] == 1, 0, train_result['EverD60CO_Mar2026'])

        return train_result


def plot_all_prediction_results(df, test_result, train_result, score_col='y_pred', bal_col='Dec2023Bal', n_deciles=10):

    train_result = add_additional_columns_for_plotting_prediction_results(df, train_result, bal_col='Dec2023Bal')
    test_result = add_additional_columns_for_plotting_prediction_results(df, test_result, bal_col='Dec2023Bal')

    agg_model_train, agg_bcq_train = plot_prediction_results(train_result, bal_col=bal_col, bad_col='EverD60CO_Sep2025', score_col=score_col, n_deciles=n_deciles, 
        title = 'Train Set (Loan Count {:,.0f}) cut off 9/30/2025\nBalance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality'.format(len(train_result)))

    agg_model_train, agg_bcq_train = plot_prediction_results(train_result, bal_col=bal_col, bad_col='EverD60CO_Mar2026', score_col=score_col, n_deciles=n_deciles, 
            title = 'Train Set (Loan Count {:,.0f}) cut off 3/31/2026\nBalance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality'.format(len(train_result)))

    agg_model_train, agg_bcq_train = plot_prediction_results(train_result, bal_col=bal_col, bad_col='new_default_flag', score_col=score_col, n_deciles=n_deciles, 
            title = 'Train Set (Loan Count {:,.0f}) New DQ60/CO between 9/30/2025 and 3/31/2026\nBalance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality'.format(len(train_result)))

    agg_model_test, agg_bcq_test = plot_prediction_results(test_result, bal_col=bal_col, bad_col='EverD60CO_Mar2026', score_col=score_col, n_deciles=n_deciles, 
            title = 'Test Set (Loan Count {:,.0f}) cut off 3/31/2026\nBalance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality'.format(len(test_result)))

    agg_model_test, agg_bcq_test = plot_prediction_results(test_result, bal_col=bal_col, bad_col='new_default_flag', score_col=score_col, n_deciles=n_deciles, 
            title = 'Test Set (Loan Count {:,.0f}) New DQ60/CO between 9/30/2025 and 3/31/2026\nBalance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality'.format(len(test_result)))

    agg_model_test, agg_bcq_test = plot_prediction_results(test_result, bal_col=bal_col, bad_col='EverD60CO_Sep2025', score_col=score_col, n_deciles=n_deciles, 
            title = 'Test Set (Loan Count {:,.0f}) cut off 9/30/2025\nBalance-Weighted EverD60CO by Decile\nModel Score vs Borrower Credit Quality'.format(len(test_result)))


def clean_bulk_tape(df_bulk_tape, weighted_quarter_rate_mapping_training):
    # Merge back to original LoanTape
    print('Duplicated Loan Count: ',
    df_bulk_tape.shape[0] - df_bulk_tape['Loan_Identifier'].nunique()
    )

    df_bulk_tape = df_bulk_tape.drop_duplicates(
        subset=['Loan_Identifier'],
        keep='first'
    )

    print('Final Loan Tape Shape: ', df_bulk_tape.shape)

    df_bulk_tape = df_bulk_tape[['Loan_Identifier', 
                                 
                                    'Borrower_Identifier',
                                    'Borrower_Credit_Quality', 
                                    'Borrower_Employment_Status',
                                    'Primary_Income', 
                                    'Geographic_Region', 
                                    'Origination_Date',
                                    'Maturity_Date', 
                                    'Original_Loan_Term', 
                                    'Remaining_Loan_Term',
                                    'Loan_Age', 
                                    'Origination_Balance', 
                                    'Current_Balance',
                                    'Scheduled_Payment_Due', 
                                    'LOANS_PURPOSE', 
                                    'Annual_Percentage_Rate',
                                    'Current_Interest_Rate', 
                                    'Number_of_Borrowers', 
                                    'Resident',
                                    'Set_Off_Amount'
                            ]]

    df_bulk_tape = add_readable_category_buckets(df_bulk_tape)
    # print('Added readable category buckets.', df_bulk_tape.columns)

    df_bulk_tape['Origination_Date'] = pd.to_datetime(df_bulk_tape['Origination_Date'], errors='coerce')

    df_bulk_tape['Origination_Quarter'] = (
        df_bulk_tape['Origination_Date'].dt.to_period('Q').astype(str)
    )

    quarter_avg_rate = (
        df_bulk_tape
        .groupby('Origination_Quarter')
        .apply(lambda g: np.average(g['Current_Interest_Rate'], 
                                    weights=g['Current_Balance']))
        .rename('weighted_quarter_rate')
        .reset_index()
    )

    quarter_avg_rate = quarter_avg_rate.merge(weighted_quarter_rate_mapping_training, 
                        on='Origination_Quarter', 
                        how='left',
                        suffixes=('_quarter_avg', '_training'))

    quarter_avg_rate['weighted_quarter_rate'] = quarter_avg_rate['weighted_quarter_rate_training'].combine_first(quarter_avg_rate['weighted_quarter_rate_quarter_avg'])

    df_bulk_tape = df_bulk_tape.merge(quarter_avg_rate, on='Origination_Quarter', how='left')

    df_bulk_tape['normalized_interest_rate'] = (
            df_bulk_tape['Current_Interest_Rate'] - df_bulk_tape['weighted_quarter_rate']
        )


    r = df_bulk_tape['Current_Interest_Rate'] / 100 / 12
    n = df_bulk_tape['Original_Loan_Term']
    bal = df_bulk_tape['Origination_Balance']

    # Monthly payment formula, with zero-rate handling
    df_bulk_tape['MonthlyPayment'] = np.where(
        r == 0,
        bal / n,                                       # zero-interest case
        r * bal / (1 - (1 + r)**(-n))                  # standard amortization
    )

    df_bulk_tape['PTI'] = df_bulk_tape['MonthlyPayment'] / df_bulk_tape['Primary_Income'] * 12 * 100

    df_bulk_tape['Has 2 Borrowers'] = np.where(df_bulk_tape['Number_of_Borrowers'] == 2, 1, 0)
    df_bulk_tape['Has 2 Borrowers x Borrower Credit Quality'] = df_bulk_tape['Has 2 Borrowers'] * df_bulk_tape['Borrower_Credit_Quality']

    df_bulk_tape['Has 1 Borrowers'] = np.where(df_bulk_tape['Number_of_Borrowers'] == 1, 1, 0)
    df_bulk_tape['Has 1 Borrowers x Borrower Credit Quality'] = df_bulk_tape['Has 1 Borrowers'] * df_bulk_tape['Borrower_Credit_Quality']

    df_bulk_tape.rename(columns={'Number_of_Borrowers': 'Number of Borrowers'}, inplace=True)
    df_bulk_tape.rename(columns={'Original_Loan_Term': 'Original_Term'}, inplace=True)
    df_bulk_tape.rename(columns={'Loan_Age': 'LoanAgeDec2023'}, inplace=True)
    df_bulk_tape.rename(columns={'Set_Off_Amount': 'Set off amount'}, inplace=True)
    return df_bulk_tape


def plot_pivot_small_multiples(
    pivot_df,
    assumption_df=None,
    metric_name="CPR",
    segments=None,
    date_col=None,
    assumption_date_col=None,
    is_percent=True,
    percent_input="decimal",   # "decimal" if 0.05 = 5%, "whole" if 5 = 5%
    moving_average_window=None,
    trim_last_n=0,
    nrows=2,
    ncols=5,
    figsize=(18, 8),
    realized_label="Realized",
    assumption_label="Assumption",
    show_legend=True,
):
    def prep_date_index(df, date_col_input=None):
        df = df.copy()

        if date_col_input is not None:
            df[date_col_input] = pd.to_datetime(df[date_col_input])
            df = df.set_index(date_col_input)
        elif "Data_Cut_Off_Date" in df.columns:
            df["Data_Cut_Off_Date"] = pd.to_datetime(df["Data_Cut_Off_Date"])
            df = df.set_index("Data_Cut_Off_Date")
        else:
            df.index = pd.to_datetime(df.index)

        return df.sort_index()

    plot_df = prep_date_index(pivot_df, date_col)

    if assumption_df is not None:
        assumption_plot_df = prep_date_index(assumption_df, assumption_date_col)
    else:
        assumption_plot_df = None

    if segments is None:
        segments = list(plot_df.columns[: nrows * ncols])

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=False
    )

    axes = np.asarray(axes).flatten()

    for ax, seg in zip(axes, segments):
        y_realized = pd.to_numeric(plot_df[seg], errors="coerce").copy()

        if moving_average_window is not None:
            y_realized = y_realized.rolling(moving_average_window, min_periods=1).mean()

        if trim_last_n and trim_last_n > 0:
            y_realized.iloc[-trim_last_n:] = np.nan

        ax.plot(
            plot_df.index.to_numpy(),
            y_realized.to_numpy(dtype=float),
            linewidth=2,
            color="#1f4e79",
            label=realized_label
        )

        if assumption_plot_df is not None and seg in assumption_plot_df.columns:
            y_assumption = pd.to_numeric(assumption_plot_df[seg], errors="coerce").copy()

            ax.plot(
                assumption_plot_df.index.to_numpy(),
                y_assumption.to_numpy(dtype=float),
                linewidth=2,
                linestyle="--",
                color="#c00000",
                label=assumption_label
            )

        ax.set_title(str(seg), fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.25)

        if is_percent:
            if percent_input == "decimal":
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            elif percent_input == "whole":
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

        if show_legend:
            ax.legend(fontsize=8, frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(segments):]:
        ax.set_visible(False)

    fig.suptitle(f"{metric_name} by Final PD Segment", fontsize=14, weight="bold", x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


def plots(df_perf, 
          performance_col, 
          grouping_col="PD_Model_Score_Decile",
          percentage = False,
          moving_average_window = 1,
          trim_last_n = 0):
    pivot = df_perf.pivot_table(
                                        index="Data_Cut_Off_Date",
                                        columns=grouping_col,
                                        values=performance_col,
                                        aggfunc="max"
                                    )
    title = performance_col + " by Final PD Segment (Worst to Best: 1.0 - 10.0)"
    if moving_average_window > 0:
        title = title + ' (Moving Average ' + str(int(moving_average_window)) + ' Months)'
    plot_finance_style(
                        pivot,
                        title=title,
                        ylabel=performance_col,
                        percentage=percentage,
                        decimal=2,
                        data_labels=False,
                        figsize=(12, 6),
                        legend_location="upper left",
                        moving_average_window=moving_average_window,
                        trim_last_n = trim_last_n

                    )
    
    return pivot