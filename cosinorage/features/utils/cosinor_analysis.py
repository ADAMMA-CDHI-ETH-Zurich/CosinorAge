###########################################################################
# Copyright (C) 2024 ETH Zurich
# CosinorAge: Prediction of biological age based on accelerometer data
# using the CosinorAge method proposed by Shim, Fleisch and Barata
#(https://www.nature.com/articles/s41746-024-01111-x)
# 
# Authors: Jacob Leo Oskar Hunecke
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

def cosinor_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    A parametric approach to study circadian rhythmicity assuming cosinor shape, fitting a model for each day.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a Timestamp index and a column 'ENMO' containing minute-level activity data.

    Returns:
    --------
    tuple:
        - pandas.DataFrame: DataFrame with columns:
            - MESOR: Midline Estimating Statistic Of Rhythm (rhythm-adjusted mean)
            - amplitude: Half the difference between maximum and minimum values
            - acrophase: Time of peak relative to midnight in radians
            - acrophase_time: Time of peak in hours (0-24)
        - pandas.DataFrame: Fitted values for each timepoint
    
    Raises:
    -------
    ValueError:
        If DataFrame doesn't have required 'ENMO' column or timestamp index
        If data length is not a multiple of 1440 (minutes in a day)
        If any day doesn't have exactly 1440 data points
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
        group['time'] = time_minutes  # Time within the day (minutes)
        
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
        acrophase_time = acrophase/(2*np.pi)*24

        if not fitted_vals_df.empty:
            fitted_vals_df = pd.concat([fitted_vals_df, model.fittedvalues], ignore_index=False)
        else:
            fitted_vals_df = pd.DataFrame(model.fittedvalues)

        if acrophase < 0:
            acrophase += 2*np.pi

        if acrophase_time < 0:
            acrophase_time += 24

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


def cosinor_multiday(df: pd.DataFrame) -> pd.DataFrame:
    """
    A parametric approach to study circadian rhythmicity assuming cosinor shape, fitting a model for multiple days.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a Timestamp index and a column 'ENMO' containing minute-level activity data.

    Returns:
    --------
    tuple:
        - dict: Dictionary containing cosinor parameters:
            - MESOR: Midline Estimating Statistic Of Rhythm (rhythm-adjusted mean)
            - amplitude: Half the difference between maximum and minimum values
            - acrophase: Time of peak relative to midnight in radians
            - acrophase_time: Time of peak in hours (0-24)
        - pandas.Series: Fitted values for each timepoint

    Raises:
    -------
    ValueError:
        If DataFrame doesn't have required 'ENMO' column or timestamp index
        If data length is not a multiple of 1440 (minutes in a day)
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

    time_minutes = np.arange(1, total_minutes + 1)
    df['time'] = time_minutes  # Time within the day (hours)
    
    # Add cosine and sine components
    df['cos'] = np.cos(2 * np.pi * df['time'] / 1440)
    df['sin'] = np.sin(2 * np.pi * df['time'] / 1440)
    
    # Fit cosinor model
    model = ols("ENMO ~ cos + sin", data=df).fit()
    
    # Extract parameters
    mesor = float(model.params['Intercept'])
    beta_cos = model.params['cos']
    beta_sin = model.params['sin']
    amplitude = float(np.sqrt(beta_cos**2 + beta_sin**2))
    acrophase = float(np.arctan2(beta_sin, beta_cos))
    acrophase_time = float(acrophase/(2*np.pi)*24)
    fitted_vals_df = model.fittedvalues

    if acrophase < 0:
        acrophase += 2*np.pi

    # Adjust acrophase time to 0-24 hours
    if acrophase_time < 0:
        acrophase_time += 24

    # Convert the results into a DataFrame
    return {'MESOR': mesor, 'amplitude': amplitude, 'acrophase': acrophase, 'acrophase_time': acrophase_time}, fitted_vals_df