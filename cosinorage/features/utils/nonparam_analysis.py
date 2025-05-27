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

import pandas as pd
import numpy as np

from typing import List

def IS(data: pd.Series) -> float:
    r"""Calculate the interdaily stability (IS) for each day separately.

    Interdaily stability quantifies the strength of coupling between the
    rest-activity rhythm and environmental zeitgebers. It compares the
    24-hour pattern across days.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and 'IS' column containing interdaily
        stability values for each day
    """
    if len(data) == 0:
        return np.nan

    data_ = data.copy()[['ENMO']]
    data_ = data_.resample('h').mean()
    data_['hour'] = data_.index.hour

    # Calculate key values
    H = 24  # Hours per day
    D = len(pd.unique(data_.index.date))  # Number of days
    z_mean = data_['ENMO'].mean()  # Overall mean
    
    # Calculate hourly means across days 
    hourly_means = data_.groupby('hour')['ENMO'].mean()
    
    # Calculate numerator
    numerator = D * np.sum(np.power(hourly_means - z_mean, 2), axis=0)
    
    # Calculate denominator
    denominator = np.sum(np.power(data_['ENMO'] - z_mean, 2), axis=0)
    
    if denominator == 0:
        return np.nan
        
    IS = float(numerator / denominator)
    
    return IS


def IV(data: pd.Series) -> float:
    r"""Calculate the intradaily variability for each day separately.

    Intradaily variability quantifies the fragmentation of rest-activity patterns
    within each 24-hour period. It is calculated as the ratio of the mean squared
    first derivative to the variance.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and 'IV' column containing intradaily
        variability values for each day
    """
    if len(data) == 0:
        return np.nan

    data_ = data.copy()[['ENMO']]
    P = len(data_)

    # resample to hourly data
    data_ = data_.resample('h').mean()
    
    # Calculate numerator: P * sum((z_p - z_{p-1})^2)
    first_derivative_squared = np.sum(np.power(data_[1:].reset_index(drop=True) - data_[:-1].reset_index(drop=True), 2), axis=0)
    numerator = float(P * first_derivative_squared.iloc[0])
    
    # Calculate denominator: (P-1) * sum((z_p - z_mean)^2)
    deviations_squared = np.sum(np.power(data_ - data_.mean(), 2), axis=0)
    denominator = float((P - 1) * deviations_squared.iloc[0])
    
    if denominator == 0:
        return np.nan
        
    IV = numerator / denominator
    
    return IV


def M10(data: pd.Series) -> List[float]:
    r"""Calculate the M10 (mean activity during the 10 most active hours) 
    and the start time of the 10 most active hours (M10_start) for each day.

    M10 provides information about the most active period during each day,
    which typically corresponds to the main activity phase.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and two columns:
        - 'M10': Mean activity during the 10 most active hours
        - 'M10_start': Hour (0-23) when the most active period starts
    """
    if len(data) == 0:
        return [], []

    data_ = data.copy()[['ENMO']]
    daily_groups = data_.groupby(data_.index.date)

    m10 = []
    m10_start = []
    for date, day_data in daily_groups:
        # calculate the rolling mean over 10-hour windows
        window_size = 600  # 10 hours * 60 minutes
        rolling_means = day_data[::-1].rolling(window=window_size, center=False).mean()[::-1].dropna()
        
        # Find the window with maximum activity
        max_mean = float(rolling_means.max().iloc[0])
        max_start_idx = rolling_means.idxmax().iloc[0]

        if pd.isna(max_mean):
            m10.append(np.nan)
            m10_start.append(np.nan)
        else:
            m10.append(max_mean)
            m10_start.append(max_start_idx)

    return m10, m10_start


def L5(data: pd.Series) -> List[float]:
    r"""Calculate the L5 (mean activity during the 5 least active hours) 
    and the start time of the 5 least active hours (L5_start) for each day.

    L5 provides information about the least active period during each day,
    which typically corresponds to the main rest phase.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and two columns:
        - 'L5': Mean activity during the 5 least active hours
        - 'L5_start': Hour (0-23) when the least active period starts
    """
    if len(data) == 0:
        return [], []

    data_ = data.copy()[['ENMO']]
    daily_groups = data_.groupby(data_.index.date)

    l5 = []
    l5_start = []
    for date, day_data in daily_groups:
        # calculate the rolling mean over 5-hour windows
        window_size = 300  # 5 hours * 60 minutes
        rolling_means = day_data[::-1].rolling(window=window_size, center=False).mean()[::-1].dropna()
        
        # Find the window with minimum activity
        min_mean = float(rolling_means.min().iloc[0])
        min_start_idx = rolling_means.idxmin().iloc[0]
    
        if pd.isna(min_mean):
            l5.append(np.nan)
            l5_start.append(np.nan)
        else:
            l5.append(min_mean)
            l5_start.append(min_start_idx)

    return l5, l5_start


def RA(m10: List[float], l5: List[float]) -> List[float]:
    r"""Calculate the relative amplitude (RA) for each day separately.

    Relative amplitude is calculated as the difference between the most active
    10-hour period and least active 5-hour period, divided by their sum.
    This provides a normalized measure of the daily activity rhythm strength.

    Parameters
    ----------
    data : pd.Series
        Time series data containing activity measurements with datetime index
        and 'ENMO' column

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and 'RA' column containing relative
        amplitude values for each day
    """
    if len(m10) == 0 or len(l5) == 0:
        return []

    if len(m10) != len(l5):
        raise ValueError("m10 and l5 must have the same length")
    
    ra = [(m10[i] - l5[i]) / (m10[i] + l5[i]) for i in range(len(m10))]
    
    return ra


