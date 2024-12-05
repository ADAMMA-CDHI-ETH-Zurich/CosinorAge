###########################################################################
# Copyright (C) 2024 ETH Zurich
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

def IV(data: pd.Series) -> pd.DataFrame:
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
        return pd.DataFrame(columns=['IV'])

    data_ = data.copy()[['ENMO']]

    daily_groups = data_.groupby(data_.index.date)
    
    # Calculate IV for each day
    daily_ivs = []
    for date, day_data in daily_groups:
        if len(day_data) <= 1:
            iv = np.nan
        else:
            c_1h = day_data.diff(1).pow(2).mean().values[0]
            d_1h = day_data.var().values[0]
            if d_1h == 0:
                iv = np.nan
            else:
                iv = (c_1h / d_1h)
        daily_ivs.append({'date': date, 'IV': iv})
    
    # Create DataFrame with results
    iv_df = pd.DataFrame(daily_ivs)
    if not iv_df.empty:
        iv_df.set_index('date', inplace=True)
    
    return iv_df


def IS(data: pd.Series) -> pd.DataFrame:
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
        return pd.DataFrame(columns=['IS'])

    data_ = data.copy()[['ENMO']]
    data_['hour'] = data_.index.hour
    data_['minute'] = data_.index.minute

    # Calculate the mean 24-hour profile
    mean_profile = data_.groupby(['hour', 'minute'])['ENMO'].mean()

    # Calculate IS for each day
    daily_groups = data_.groupby(data_.index.date)
    daily_iss = []
    for date, day_data in daily_groups:
        # Match the day's data with the 24-hour mean profile
        day_data['hour_minute'] = list(zip(day_data['hour'], day_data['minute']))
        day_profile = day_data['hour_minute'].map(mean_profile)

        # Compute IS
        mean_daily_value = day_data['ENMO'].mean()
        numerator = ((day_profile - mean_daily_value) ** 2).sum()
        denominator = ((day_data['ENMO'] - mean_daily_value) ** 2).sum()
        is_value = numerator / denominator if denominator != 0 else 0

        daily_iss.append({'date': date, 'IS': is_value})

    # Create DataFrame with results
    is_df = pd.DataFrame(daily_iss)
    is_df.set_index('date', inplace=True)

    return is_df


def RA(data: pd.Series) -> pd.DataFrame:
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
    if len(data) == 0:
        return pd.DataFrame(columns=['RA'])

    data_ = data.copy()[['ENMO']]
    data_['hour'] = data_.index.hour

    # Group data by day
    daily_groups = data_.groupby(data_.index.date)

    # Calculate RA for each day
    daily_ras = []
    for date, day_data in daily_groups:
        # Group data by hour within the day
        hourly_means = day_data.groupby('hour')['ENMO'].mean()

        # Find most active 10 hours and least active 5 hours
        top_10h = hourly_means.nlargest(10).mean()  # Mean activity of the most active 10 hours
        bottom_5h = hourly_means.nsmallest(5).mean()  # Mean activity of the least active 5 hours

        # Compute RA
        ra_value = (top_10h - bottom_5h) / (top_10h + bottom_5h) if (top_10h + bottom_5h) != 0 else 0

        daily_ras.append({'date': date, 'RA': ra_value})

    # Create DataFrame with results
    ra_df = pd.DataFrame(daily_ras)
    ra_df.set_index('date', inplace=True)

    return ra_df


def M10(data: pd.Series) -> pd.DataFrame:
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
        return pd.DataFrame(columns=['M10', 'M10_start'])

    data_ = data.copy()[['ENMO']]
    data_['hour'] = data_.index.hour

    # Group data by day
    daily_groups = data_.groupby(data_.index.date)

    # Calculate M10 and M10_start for each day
    daily_m10 = []
    for date, day_data in daily_groups:
        # Group data by hour within the day
        hourly_means = day_data.groupby('hour')['ENMO'].mean()

        # Find most active 10 hours
        top_10h = hourly_means.nlargest(10)
        top_10h_mean = top_10h.mean()
        m10_start_hour = top_10h.idxmax()  # The hour with the highest activity in the top 10

        daily_m10.append({'date': date, 'M10': top_10h_mean, 'M10_start': m10_start_hour})

    # Create DataFrame with results
    m10_df = pd.DataFrame(daily_m10)
    m10_df.set_index('date', inplace=True)

    return m10_df


def L5(data: pd.Series) -> pd.DataFrame:
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
        return pd.DataFrame(columns=['L5', 'L5_start'])

    data_ = data.copy()[['ENMO']]
    data_['hour'] = data_.index.hour

    # Group data by day
    daily_groups = data_.groupby(data_.index.date)

    # Calculate L5 and L5_start for each day
    daily_l5 = []
    for date, day_data in daily_groups:
        # Group data by hour within the day
        hourly_means = day_data.groupby('hour')['ENMO'].mean()

        # Find least active 5 hours
        bottom_5h = hourly_means.nsmallest(5)
        bottom_5h_mean = bottom_5h.mean()
        l5_start_hour = bottom_5h.idxmin()  # The hour with the lowest activity in the bottom 5

        daily_l5.append({'date': date, 'L5': bottom_5h_mean, 'L5_start': l5_start_hour})

    # Create DataFrame with results
    l5_df = pd.DataFrame(daily_l5)
    l5_df.set_index('date', inplace=True)
    return l5_df

