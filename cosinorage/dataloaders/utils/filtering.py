import pandas as pd
import numpy as np
from typing import List
from datetime import datetime, timedelta


def filter_incomplete_days(df: pd.DataFrame, data_freq: float) -> pd.DataFrame:
    """
    Filter out data from incomplete days to ensure 24-hour data periods.

    This function removes data from the first and last days in the DataFrame
    to ensure that only complete 24-hour data is retained.

    Args:
        data_all (pd.DataFrame): DataFrame with a 'TIMESTAMP' column in datetime
            format, which is used to determine the day.

    Returns:
        pd.DataFrame: Filtered DataFrame excluding the first and last days.
        If there
        are fewer than two unique dates in the data, an empty DataFrame is
        returned.
    """

    # Filter out incomplete days
    try:
        # Calculate expected number of data points for a full 24-hour day
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
    Return data frame containing only consecutive days.
    """
    days = np.unique(df.index.date)
    days = largest_consecutive_sequence(days)

    if len(days) < 4:
        raise ValueError("Less than 4 consecutive days found")

    df = df[pd.Index(df.index.date).isin(days)]
    return df


def largest_consecutive_sequence(dates: List[datetime]) -> List[datetime]:
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