import os
import numpy as np
import pandas as pd
from glob import glob
from typing import Tuple, Optional, Union, Any
from tqdm import tqdm


def read_acc_csvs(directory_path: str) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Concatenate all CSV files in a directory into a single DataFrame.

    This function reads all CSV files in the specified directory that match the
    '*.sensor.csv' pattern, concatenates them, and returns a single DataFrame
    containing only the 'HEADER_TIMESTAMP', 'X', 'Y', and 'Z' columns.

    Args:
        directory_path (str): Path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing the accelerometer
        data from all CSV files, with columns 'HEADER_TIMESTAMP', 'X', 'Y',
        'Z', sorted by 'HEADER_TIMESTAMP'.
    """
    file_names = glob(os.path.join(directory_path, "*.sensor.csv"))

    if not file_names:
        print(f"No files found in {directory_path}")
        return pd.DataFrame(), None

    # Read all CSV files and concatenate into a single DataFrame
    data_frames = []
    try:
        for file in tqdm(file_names, desc="Loading CSV files"):
            try:
                df = pd.read_csv(file,
                                 usecols=['HEADER_TIMESTAMP', 'X', 'Y', 'Z'])
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        data = pd.concat(data_frames, ignore_index=True)
    except Exception as e:
        print(f"Error concatenating CSV files: {e}")
        return pd.DataFrame(), None

    # Convert timestamps to datetime format
    try:
        data['HEADER_TIMESTAMP'] = pd.to_datetime(data['HEADER_TIMESTAMP'])
        data = data.sort_values(by='HEADER_TIMESTAMP')
        data.rename(columns={'HEADER_TIMESTAMP': 'TIMESTAMP'}, inplace=True)
    except Exception as e:
        print(f"Error converting timestamps: {e}")
        return pd.DataFrame(), None

    # check if timestamp frequency is consistent up to 1ms
    time_diffs = data['TIMESTAMP'].diff().dt.round('1ms')
    unique_diffs = time_diffs.unique()
    if (not len(unique_diffs) == 1) and (
            not (len(unique_diffs) == 2) and unique_diffs[0] - unique_diffs[
        1] <= pd.Timedelta('1ms')):
        raise ValueError("Inconsistent timestamp frequency detected.")

    # resample timestamps with mean frequency
    sample_rate = 1 / unique_diffs.mean().total_seconds()
    timestamps = data['TIMESTAMP']
    start_timestamp = pd.to_datetime(timestamps.iloc[0])
    time_deltas = pd.to_timedelta(np.arange(len(timestamps)) / sample_rate,
                                  unit='s')
    data['TIMESTAMP'] = start_timestamp + time_deltas

    # determine frequency in Hz of accelerometer data
    time_diffs = data['TIMESTAMP'].diff().dropna()
    acc_freq = 1 / time_diffs.mean().total_seconds()

    # set timestamp as index
    data.set_index('TIMESTAMP', inplace=True)

    return data, acc_freq
    #return data[['TIMESTAMP', 'X', 'Y', 'Z']], acc_freq


def read_enmo_csv(file_path: str, source: str) -> Union[pd.DataFrame, tuple[Any, Union[float, Any]]]:
    # based on data doc_source file format might look different
    if source == 'uk-biobank':
        time_col = 'time'
        enmo_col = 'ENMO_t'
    else:
        raise ValueError(
            "Invalid doc_source specified. Please specify, e.g., 'uk-biobank'.")

    # Read the CSV file
    try:
        data = pd.read_csv(file_path)[[time_col, enmo_col]]
        data = data.sort_values(by=time_col)
        data.rename(columns={enmo_col: 'ENMO'}, inplace=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

    # Convert timestamps to datetime format
    try:
        data[time_col] = pd.to_datetime(data[time_col], format='mixed')
        data.rename(columns={time_col: 'TIMESTAMP'}, inplace=True)
    except Exception as e:
        print(f"Error converting timestamps: {e}")
        return pd.DataFrame()

    # check if timestamp frequency is consistent up to 1ms
    time_diffs = data['TIMESTAMP'].diff().dropna()
    unique_diffs = time_diffs.unique()
    if not len(unique_diffs) == 1:
        raise ValueError("Inconsistent timestamp frequency detected.")

    return data[['TIMESTAMP', 'ENMO']]


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
