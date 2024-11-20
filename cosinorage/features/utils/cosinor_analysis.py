import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

def act_cosinor(df, window=1, export_ts=True):
    """
    A parametric approach to study circadian rhythmicity assuming cosinor shape.

    Parameters:
    df : pandas.DataFrame
        DataFrame with a time index and a column 'ENMO' containing minute-level activity data.
        The 'time' column is expected to be a datetime-like object.
    window : int
        The window size of the data (e.g., window=1 means each epoch is 1 minute).
    export_ts : bool
        Whether to export time series data.

    Returns:
    dict:
        Contains MESOR, amplitude, acrophase, acrophase time (hours), number of days, and optionally the time series.
    """
    # Ensure the DataFrame contains the required columns
    if 'ENMO' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame must have a datetime-like index and an 'ENMO' column.")

    df_ = df.copy()

    # Check that data length matches 1440-minute cycles
    total_minutes = len(df_)
    dim = 1440 // window
    if total_minutes % dim != 0:
        raise ValueError("Data length is not a multiple of a day (1440 minutes or adjusted for the window size).")

    n_days = total_minutes // dim

    # Prepare time variable for modeling
    time_minutes = np.arange(1, total_minutes + 1) / (60 / window)
    df_['time'] = time_minutes % 24  # Time within the day (hours)

    # Add cosine and sine components
    df_['cos'] = np.cos(2 * np.pi * df_['time'] / 24)
    df_['sin'] = np.sin(2 * np.pi * df_['time'] / 24)

    # Fit cosinor model
    model = ols("ENMO ~ cos + sin", data=df_).fit()

    # Extract parameters
    mesor = model.params['Intercept']
    beta_cos = model.params['cos']
    beta_sin = model.params['sin']
    amplitude = np.sqrt(beta_cos**2 + beta_sin**2)
    acrophase = np.arctan2(-beta_sin, beta_cos)
    acrophase_time = (-acrophase * 24) / (2 * np.pi)

    # Adjust acrophase time to 0-24 hours
    if acrophase_time < 0:
        acrophase_time += 24

    if export_ts:
        # Create a fitted time series
        df_['fitted'] = model.fittedvalues
        time_across_days = df_['time'].copy()
        drops = np.where(np.diff(df_['time']) < 0)[0] + 1
        for k in drops:
            time_across_days.iloc[k:] += 24
        df_['time_across_days'] = time_across_days
        cosinor_ts = df_[['ENMO', 'fitted']]
    else:
        cosinor_ts = None

    # Prepare output
    params = {
        "MESOR": float(mesor),
        "amp": float(amplitude),
        "acr": float(acrophase),
        "acrotime": float(acrophase_time),
        "ndays": n_days
    }

    return {"params": params, "cosinor_ts": cosinor_ts}