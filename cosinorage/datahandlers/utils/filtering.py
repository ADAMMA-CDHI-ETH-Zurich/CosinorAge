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

from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


def filter_incomplete_days(
    df: pd.DataFrame,
    data_freq: float,
    expected_points_per_day: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filter out data from incomplete days to ensure 24-hour data periods.

    This function removes data from days that don't have the expected number of data points
    to ensure that only complete 24-hour data is retained for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index, which is used to determine the day.
        The index should contain datetime objects.
    data_freq : float
        Frequency of data collection in Hz (e.g., 1/60 for minute-level data).
    expected_points_per_day : int, optional
        Expected number of data points per day. If None, calculated using data_freq * 86400.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only complete days.
        Returns empty DataFrame if an error occurs during processing.

    Notes
    -----
    - Calculates expected points per day as data_freq * 60 * 60 * 24 if not provided
    - Groups data by date and counts points per day
    - Retains only days with sufficient data points
    - Removes the temporary 'DATE' column before returning
    - Handles errors gracefully by returning empty DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample data with some incomplete days
    >>> dates = pd.date_range('2023-01-01', periods=5000, freq='min')
    >>> data = pd.DataFrame({'value': np.random.randn(5000)}, index=dates)
    >>>
    >>> # Filter incomplete days (expecting 1440 points per day for minute data)
    >>> filtered_data = filter_incomplete_days(data, data_freq=1/60, expected_points_per_day=1440)
    >>> print(f"Original days: {len(data.index.date.unique())}")
    >>> print(f"Complete days: {len(filtered_data.index.date.unique())}")
    """

    # Filter out incomplete days
    try:
        # Calculate expected number of data points for a full 24-hour day
        if expected_points_per_day == None:
            expected_points_per_day = data_freq * 60 * 60 * 24

        # Extract the date from each timestamp
        _df = df.copy()
        # timestamp is index
        _df["DATE"] = _df.index.date

        # Count data points for each day
        daily_counts = _df.groupby("DATE").size()

        # Identify complete days based on expected number of data points
        complete_days = daily_counts[
            daily_counts >= expected_points_per_day
        ].index

        # Filter the DataFrame to include only rows from complete days
        filtered_df = _df[_df["DATE"].isin(complete_days)]

        # Drop the helper 'DATE' column before returning
        return filtered_df.drop(columns=["DATE"])

    except Exception as e:
        print(f"Error filtering incomplete days: {e}")
        return pd.DataFrame()


def filter_consecutive_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to retain only the longest sequence of consecutive days.

    This function identifies the longest sequence of consecutive days in the data
    and filters the DataFrame to include only those days. This is important for
    circadian rhythm analysis which requires continuous data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index containing the data to filter.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only the longest sequence of consecutive days.

    Raises
    ------
    ValueError
        If less than 2 consecutive days are found in the data.

    Notes
    -----
    - Extracts unique dates from the datetime index
    - Finds the longest consecutive sequence using largest_consecutive_sequence
    - Requires at least 2 consecutive days for valid analysis
    - Filters the DataFrame to include only data from consecutive days
    - Important for circadian rhythm analysis which requires continuous data

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample data with gaps
    >>> dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03',
    ...                         '2023-01-05', '2023-01-06', '2023-01-07'])
    >>> data = pd.DataFrame({'value': np.random.randn(len(dates))}, index=dates)
    >>>
    >>> # Filter to longest consecutive sequence
    >>> filtered_data = filter_consecutive_days(data)
    >>> print(f"Original dates: {data.index.date.tolist()}")
    >>> print(f"Consecutive dates: {filtered_data.index.date.tolist()}")
    """
    days = np.unique(df.index.date)
    days = largest_consecutive_sequence(days)

    if len(days) < 1:
        raise ValueError("Less than 1 day found")

    df = df[pd.Index(df.index.date).isin(days)]
    return df


def largest_consecutive_sequence(dates: List[datetime]) -> List[datetime]:
    """
    Find the longest sequence of consecutive dates in a list.

    This function analyzes a list of dates and returns the longest subsequence
    of consecutive dates. It's used to identify continuous periods of data
    for circadian rhythm analysis.

    Parameters
    ----------
    dates : List[datetime]
        List of dates to analyze for consecutive sequences.

    Returns
    -------
    List[datetime]
        Longest sequence of consecutive dates found.
        Returns empty list if input is empty.

    Notes
    -----
    - Sorts and removes duplicate dates before processing
    - Compares dates using timedelta(days=1) for consecutive day detection
    - Maintains the original order within consecutive sequences
    - Handles edge cases like empty lists and single dates
    - Used internally by filter_consecutive_days

    Examples
    --------
    >>> from datetime import datetime
    >>>
    >>> # Example with gaps in dates
    >>> dates = [
    ...     datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3),
    ...     datetime(2023, 1, 5), datetime(2023, 1, 6), datetime(2023, 1, 7)
    ... ]
    >>> consecutive = largest_consecutive_sequence(dates)
    >>> print(f"Longest consecutive sequence: {consecutive}")
    >>> # Output: [datetime(2023, 1, 5), datetime(2023, 1, 6), datetime(2023, 1, 7)]
    >>>
    >>> # Example with single date
    >>> single_date = [datetime(2023, 1, 1)]
    >>> result = largest_consecutive_sequence(single_date)
    >>> print(f"Single date result: {result}")
    >>> # Output: [datetime(2023, 1, 1)]
    """
    if len(dates) == 0:  # Handle empty list
        return []

    # Sort and remove duplicates
    dates = sorted(set(dates))
    longest_sequence = []
    current_sequence = [dates[0]]

    for i in range(1, len(dates)):
        if dates[i] - dates[i - 1] == timedelta(
            days=1
        ):  # Check for consecutive days
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
