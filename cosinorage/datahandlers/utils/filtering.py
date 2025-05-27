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
from datetime import datetime, timedelta


def filter_incomplete_days(df: pd.DataFrame, data_freq: float, expected_points_per_day: int = None) -> pd.DataFrame:
    """
    Filter out data from incomplete days to ensure 24-hour data periods.

    This function removes data from days that don't have the expected number of data points
    to ensure that only complete 24-hour data is retained.

    Args:
        df (pd.DataFrame): DataFrame with datetime index, which is used to determine the day
        data_freq (float): Frequency of data collection in Hz
        expected_points_per_day (int, optional): Expected number of data points per day. 
            If None, calculated using data_freq

    Returns:
        pd.DataFrame: Filtered DataFrame containing only complete days.
            Returns empty DataFrame if an error occurs during processing.

    Raises:
        Exception: Prints error message and returns empty DataFrame if processing fails
    """

    # Filter out incomplete days
    try:
        # Calculate expected number of data points for a full 24-hour day
        if expected_points_per_day == None:
            expected_points_per_day = data_freq * 60 * 60 * 24

        # Extract the date from each timestamp
        _df = df.copy()
        # timestamp is index
        _df['DATE'] = _df.index.date

        # Count data points for each day
        daily_counts = _df.groupby('DATE').size()

        # Identify complete days based on expected number of data points
        complete_days = daily_counts[
            daily_counts >= expected_points_per_day].index

        # Filter the DataFrame to include only rows from complete days
        filtered_df = _df[_df['DATE'].isin(complete_days)]

        # Drop the helper 'DATE' column before returning
        return filtered_df.drop(columns=['DATE'])

    except Exception as e:
        print(f"Error filtering incomplete days: {e}")
        return pd.DataFrame()


def filter_consecutive_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to retain only the longest sequence of consecutive days.

    Args:
        df (pd.DataFrame): DataFrame with datetime index

    Returns:
        pd.DataFrame: Filtered DataFrame containing only consecutive days

    Raises:
        ValueError: If less than 4 consecutive days are found in the data
    """
    days = np.unique(df.index.date)
    days = largest_consecutive_sequence(days)

    if len(days) < 4:
        raise ValueError("Less than 4 consecutive days found")

    df = df[pd.Index(df.index.date).isin(days)]
    return df


def largest_consecutive_sequence(dates: List[datetime]) -> List[datetime]:
    """
    Find the longest sequence of consecutive dates in a list.

    Args:
        dates (List[datetime]): List of dates to analyze

    Returns:
        List[datetime]: Longest sequence of consecutive dates found.
            Returns empty list if input is empty.

    Example:
        >>> dates = [datetime(2023,1,1), datetime(2023,1,2), datetime(2023,1,4)]
        >>> largest_consecutive_sequence(dates)
        [datetime(2023,1,1), datetime(2023,1,2)]
    """
    if len(dates) == 0:  # Handle empty list
        return []
    
    # Sort and remove duplicates
    dates = sorted(set(dates))
    longest_sequence = []
    current_sequence = [dates[0]]
    
    for i in range(1, len(dates)):
        if dates[i] - dates[i - 1] == timedelta(days=1):  # Check for consecutive days
            current_sequence.append(dates[i])
        else:
            # Update longest sequence if current is longer
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = [dates[i]]  # Start a new sequence
    
    # Final check after loop
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence
    
    return longest_sequence