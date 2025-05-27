###########################################################################
# Copyright (C) 2025 ETH Zurich
# CosinorAge: Prediction of biological age based on accelerometer data
# using the CosinorAge method proposed by Shim, Fleisch and Barata
# (https://www.nature.com/articles/s41746-024-01111-x)
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

import numpy as np
import pandas as pd


def calculate_enmo(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Calculate the Euclidean Norm Minus One (ENMO) metric from accelerometer data.

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with columns
            'X', 'Y', and 'Z' for accelerometer readings along the three axes.
        verbose (bool, optional): If True, prints processing information. Defaults to False.

    Returns:
        numpy.ndarray: Array of ENMO values. Values are truncated at 0, meaning negative
            values are set to 0. Returns np.nan if calculation fails.

    Notes:
        ENMO is calculated as the Euclidean norm of the acceleration vector minus one
        gravity unit. This metric is commonly used in physical activity research to
        quantify acceleration while accounting for gravity.
    """

    if data.empty:
        return pd.DataFrame()

    try:
        _acc_vectors = data[['X', 'Y', 'Z']].values
        _enmo_vals = np.linalg.norm(_acc_vectors, axis=1) - 1
        _enmo_vals = np.maximum(_enmo_vals, 0)
    except Exception as e:
        print(f"Error calculating ENMO: {e}")
        _enmo_vals = np.nan

    if verbose:
        print(f"Calculated ENMO for {data.shape[0]} accelerometer records")

    return _enmo_vals


def calculate_minute_level_enmo(data: pd.DataFrame, sf: float, verbose: bool = False) -> pd.DataFrame:
    """
    Resample high-frequency ENMO data to minute-level by averaging over each minute.

    Args:
        data (pd.DataFrame): DataFrame with 'TIMESTAMP' as index and 'ENMO' column 
            containing high-frequency ENMO data. Optional 'wear' column for wear time.
        sf (float): Sampling frequency of the data in Hz (samples per second).
        verbose (bool, optional): If True, prints processing information. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing minute-level aggregated data with:
            - 'ENMO': Mean ENMO value for each minute
            - 'wear': Mean wear time for each minute (if wear column exists in input)
            Index is datetime at minute resolution.

    Raises:
        ValueError: If sampling frequency is less than 1/60 Hz (less than one sample per minute).

    Notes:
        The function resamples the data to minute-level using mean aggregation.
        Timestamps are converted to datetime format in the output.
    """

    if sf < 1/60:
        raise ValueError("Sampling frequency must be at least 1 minute")

    if data.empty:
        return pd.DataFrame()

    try:    
        minute_level_enmo_df = data['ENMO'].resample('min').mean().to_frame(name='ENMO')
        # check if data has a wear column
        if 'wear' in data.columns:
            minute_level_enmo_df['wear'] = data['wear'].resample('min').mean()
        
    except Exception as e:
        print(f"Error resampling ENMO data: {e}")
        minute_level_enmo_df = pd.DataFrame()

    minute_level_enmo_df.index = pd.to_datetime(minute_level_enmo_df.index)
    
    if verbose:
        print(f"Aggregated ENMO values at the minute level leading to {minute_level_enmo_df.shape[0]} records")

    return minute_level_enmo_df
