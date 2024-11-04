import pandas as pd
import os
from datetime import timedelta
import numpy as np
from glob import glob

def concatenate_csv(directory_path):
    """
    Concatenate all CSV files in a directory into a single DataFrame.
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


def get_posix_timestamps(timestamps, sample_rate=80):
    """
    Add a POSIX timestamp column named TIMESTAMP at a given sampling rate.

    :param data_all: DataFrame with a 'HEADER_TIMESTAMP' column containing the start timestamp.
    :param sample_rate: Sampling rate in Hz (samples per second).
    :return: DataFrame with a 'TIMESTAMP' column and 'X', 'Y', 'Z' columns.
    """
    # Convert the first timestamp to datetime
    start_timestamp = pd.to_datetime(timestamps.iloc[0])

    # Generate a timedelta Series for efficient timestamp generation
    time_deltas = pd.to_timedelta(np.arange(len(timestamps)) / sample_rate, unit='s')

    # Add the timedelta Series to the start timestamp to create the full TIMESTAMP column
    posix_timestamps = start_timestamp + time_deltas

    return posix_timestamps


def filter_incomplete_days(data_all):
    """
    Remove data from the first and last day to ensure complete 24-hour data.
    """
    data_all['DATE'] = data_all['TIMESTAMP'].dt.date
    unique_dates = data_all['DATE'].unique()

    if len(unique_dates) <= 2:
        return pd.DataFrame()  # Not enough data to exclude first/last days

    # Exclude first and last days
    return data_all[(data_all['DATE'] != unique_dates[0]) &
                    (data_all['DATE'] != unique_dates[-1])]