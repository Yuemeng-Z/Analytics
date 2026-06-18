"""
Project Driver / Recreation Vehicles deal-specific helper functions.

Keep reusable deal quirks, curve extrapolation, and field-specific cleanup here.
Generic pipeline utilities belong in loanPipelineHelpers.py and shared notebook
plotting/formatting belongs in parentHelpers.py.
"""

import numpy as np
import pandas as pd

from loanPipelineHelpers import trim_last_n


def calc_payment_from_current_balance(balance, r, remaining_term):
    if r == 0:
        return balance / remaining_term

    pmt = balance * (r / (1 - (1 + r) ** (-remaining_term)))
    return pmt


def calc_sched_prin(r, balance, pmt):
    interest = balance * r
    principal = pmt - interest
    return principal


def next_balance(balance, r, remaining_term):
    pmt = calc_payment_from_current_balance(balance, r, remaining_term)
    principal = calc_sched_prin(r, balance, pmt)
    next_bal = balance - principal
    return next_bal


def calc_prepayment_amt(row):
    if row['Term'] > row['MOB'] and row['DaysDelinquent'] <= 0:
        bom_bal = row['BOM Bal']
        next_balance_calc = next_balance(
            bom_bal,
            row['ContractRate'] / 12,
            row['Term'] - row['MOB'] + 1,
        )
        calc_prepayment_amt = -(
            bom_bal
            - next_balance_calc
            - row['BV Calc Principal Payment Amt']
            - row['Default']
        )
    else:
        calc_prepayment_amt = 0

    return calc_prepayment_amt


def remove_last_n(cnl_pivot):
    cnl_pivot.loc[:, :'2021'] = cnl_pivot.loc[:, :'2021'].apply(trim_last_n, n=12)
    cnl_pivot.loc[:, '2022Q1':] = cnl_pivot.loc[:, '2022Q1':].apply(trim_last_n, n=3)
    return cnl_pivot


def last_n_month_average(arr_, last_n_average, term):
    arr = arr_[arr_ > 0]
    average = np.mean(arr[-last_n_average:])
    res = np.concatenate([
        np.repeat(float('nan'), len(arr)),
        np.repeat(average, term - len(arr)),
    ])
    month_average = int(np.mean(np.arange(len(arr) - last_n_average, len(arr))))

    return res, month_average, average


def create_extrapolated_curves_v1(df, last_n_average=6, term=174):
    count = 0
    for col in df.columns:
        res_arr, month_average, average = last_n_month_average(
            np.array(df[col]),
            last_n_average=last_n_average,
            term=term,
        )

        if count == 0:
            res = res_arr[:, np.newaxis]
        else:
            res = np.concatenate([res, res_arr[:, np.newaxis]], axis=1)

        count += 1

    res_df = pd.DataFrame(
        res,
        columns=['Extrapolated {}'.format(x) for x in df.columns],
        index=np.arange(len(res)),
    )
    return res_df


def extend_base_on_baseline(baseline_curves, arr_, last_n_average=6, term=174):
    res_arr, month_average, average = last_n_month_average(
        arr_,
        last_n_average=last_n_average,
        term=term,
    )
    multiple = average / baseline_curves[month_average]
    extrapolated_curve = np.concatenate([
        np.repeat(float('nan'), len(arr_[arr_ > 0])),
        multiple * baseline_curves[len(arr_[arr_ > 0]):],
    ])

    return extrapolated_curve, multiple


def create_extrapolated_curves(baseline_curves, df, last_n_average=6, term=174):
    count = 0
    for col in df.columns:
        extrapolated_curve, multiple = extend_base_on_baseline(
            baseline_curves,
            np.array(df[col]),
            last_n_average=last_n_average,
            term=term,
        )

        if count == 0:
            res = extrapolated_curve[:, np.newaxis]
        else:
            res = np.concatenate([res, extrapolated_curve[:, np.newaxis]], axis=1)

        count += 1

    res_df = pd.DataFrame(
        res,
        columns=['Extrapolated {}'.format(x) for x in df.columns],
        index=np.arange(len(res)),
    )
    return res_df


def strip_trailing_nans(arr):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Only works for 1D arrays")

    valid = np.where(~np.isnan(arr))[0]
    if len(valid) == 0:
        return np.array([])

    return arr[:valid[-1] + 1]


def replace_ltv(ever_dq_co_30_mob3):
    mask = (
        np.abs(
            ever_dq_co_30_mob3['Origination LTV ratio']
            - ever_dq_co_30_mob3['AmountFinanced']
        )
        < 100
    )
    ever_dq_co_30_mob3.loc[
        ever_dq_co_30_mob3.index[mask],
        'Origination LTV ratio',
    ] = ever_dq_co_30_mob3[mask].apply(
        lambda x: x['AmountFinanced'] / x['Origination LTV ratio'],
        axis=1,
    )

    mask_2 = ever_dq_co_30_mob3['Origination LTV ratio'] > 3
    ever_dq_co_30_mob3.loc[
        ever_dq_co_30_mob3.index[mask_2],
        'Origination LTV ratio',
    ] = float('nan')

    return ever_dq_co_30_mob3
