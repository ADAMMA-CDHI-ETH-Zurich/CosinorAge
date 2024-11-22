import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

def cosinor(df: pd.DataFrame) -> pd.DataFrame:
    """
    A parametric approach to study circadian rhythmicity assuming cosinor shape, fitting a model for each day.

    Parameters:
    df : pandas.DataFrame
        DataFrame with a Timestamp index and a column 'ENMO' containing minute-level activity data.
    window : int
        The window size of the data (e.g., window=1 means each epoch is 1 minute).
    export_ts : bool
        Whether to export time series data for each day.

    Returns:
    pandas.DataFrame:
        DataFrame with columns MESOR, amplitude, acrophase, acrophase time for each day.
    """
    # Ensure the DataFrame contains the required columns
    if 'ENMO' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame must have a Timestamp index and an 'ENMO' column.")

    # Ensure the data length is consistent
    total_minutes = len(df)
    dim = 1440  # Number of data points in a day
    if total_minutes % dim != 0:
        raise ValueError("Data length is not a multiple of a day (1440 minutes or adjusted for the window size).")

    # Group data by day
    df['date'] = df.index.date
    grouped = df.groupby('date')


    params = []
    fitted_vals_df = pd.DataFrame()
    for date, group in grouped:
        if len(group) != dim:
            raise ValueError(f"Day {date} does not have the expected number of data points ({dim}).")
        
        # Prepare time variable for modeling
        time_minutes = np.arange(1, len(group) + 1)
        group['time'] = time_minutes  # Time within the day (hours)
        
        # Add cosine and sine components
        group['cos'] = np.cos(2 * np.pi * group['time'] / 1440)
        group['sin'] = np.sin(2 * np.pi * group['time'] / 1440)
        
        # Fit cosinor model
        model = ols("ENMO ~ cos + sin", data=group).fit()
        
        # Extract parameters
        mesor = model.params['Intercept']
        beta_cos = model.params['cos']
        beta_sin = model.params['sin']
        amplitude = np.sqrt(beta_cos**2 + beta_sin**2)
        acrophase = np.arctan2(beta_sin, beta_cos)
        acrophase_time = acrophase/(2*np.pi)*1440

        fitted_vals_df = pd.concat([fitted_vals_df, model.fittedvalues], ignore_index=False)

        # Adjust acrophase time to 0-24 hours
        if acrophase_time < 0:
            acrophase_time += 1440

        # Append the day's results to the list
        params.append({
            "date": date,
            "MESOR": float(mesor),
            "amplitude": float(amplitude),
            "acrophase": float(acrophase),
            "acrophase_time": float(acrophase_time),
        })

    # Convert the results into a DataFrame
    params_df = pd.DataFrame(params).set_index("date")
    return params_df, fitted_vals_df
