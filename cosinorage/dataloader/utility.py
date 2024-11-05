import pandas as pd
import os
import numpy as np
from glob import glob


def concatenate_csv(directory_path: str) -> pd.DataFrame:
    """
    Concatenate all CSV files in a directory into a single DataFrame.

    This function reads all CSV files in the specified directory that match the
    '*.sensor.csv' pattern, concatenates them, and returns a single DataFrame
    containing only the 'HEADER_TIMESTAMP', 'X', 'Y', and 'Z' columns.

    Args:
        directory_path (str): Path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing the accelerometer data
        from all CSV files, with columns 'HEADER_TIMESTAMP', 'X', 'Y', 'Z',
        sorted by 'HEADER_TIMESTAMP'.
    """
    file_names = glob(os.path.join(directory_path, "*.sensor.csv"))

    if not file_names:
        print(f"No files found in {directory_path}")
        return pd.DataFrame()

    data_frames = [pd.read_csv(file) for file in file_names]
    data_all = pd.concat(data_frames, ignore_index=True)
    data_all = data_all[['HEADER_TIMESTAMP', 'X', 'Y', 'Z']]
    data_all = data_all.sort_values(by='HEADER_TIMESTAMP')

    return data_all


def get_posix_timestamps(timestamps: pd.Series, sample_rate=80) -> pd.Series:
    """
    Generate a POSIX timestamp series based on an initial timestamp and sample rate.

    This function creates a series of POSIX timestamps by adding a time delta
    at the specified sampling rate to the initial timestamp in the series.
    This is useful for creating timestamped data at a consistent sampling rate.

    Args:
        timestamps (pd.Series): Series containing a single initial timestamp.
        sample_rate (int, optional): Sampling rate in Hz (samples per second).
                                     Defaults to 80 Hz.

    Returns:
        pd.Series: Series of POSIX timestamps at the given sample rate.
    """
    start_timestamp = pd.to_datetime(timestamps.iloc[0])
    time_deltas = pd.to_timedelta(np.arange(len(timestamps)) / sample_rate, unit='s')
    posix_timestamps = start_timestamp + time_deltas

    return posix_timestamps


def filter_incomplete_days(data_all: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out data from incomplete days to ensure 24-hour data periods.

    This function removes data from the first and last days in the DataFrame
    to ensure that only complete 24-hour data is retained.

    Args:
        data_all (pd.DataFrame): DataFrame with a 'TIMESTAMP' column in datetime
            format, which is used to determine the day.

    Returns:
        pd.DataFrame: Filtered DataFrame excluding the first and last days. If there
        are fewer than two unique dates in the data, an empty DataFrame is returned.
    """
    data_all['DATE'] = data_all['TIMESTAMP'].dt.date
    unique_dates = data_all['DATE'].unique()

    if len(unique_dates) <= 2:
        return pd.DataFrame()  # Not enough data to exclude first/last days

    return data_all[(data_all['DATE'] != unique_dates[0]) &
                    (data_all['DATE'] != unique_dates[-1])]