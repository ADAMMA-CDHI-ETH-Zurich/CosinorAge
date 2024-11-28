import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Union
from glob import glob
from skdh.preprocessing import CountWearDetection


def read_smartwatch_data(directory_path: str, meta_dict: dict = {}) -> Tuple[pd.DataFrame, Optional[float]]:
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
    time_diffs = data['TIMESTAMP'].diff().dropna().dt.round('1ms')
    unique_diffs = time_diffs.unique()
    if (not len(unique_diffs) == 1) and (not (len(unique_diffs) == 2 and unique_diffs[0] - unique_diffs[1]) <= pd.Timedelta('1ms')):
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
    meta_dict.update({'raw_data_frequency': acc_freq})

    # set timestamp as index
    data.set_index('TIMESTAMP', inplace=True)

    return data


def preprocess_smartwatch_data(df: pd.DataFrame, sf: float, meta_dict: dict, preprocess_args: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Preprocess smartwatch data by performing auto-calibration, noise removal, and wear detection.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        sf (float): Sampling frequency of the accelerometer data in Hz.
        meta_dict (dict): Dictionary to store metadata such as total time, wear time, and non-wear time.
        epoch_size (int): Epoch size for calibration in seconds (default is 10).
        max_iter (int): Maximum number of iterations for auto-calibration (default is 1000).
        tol (float): Tolerance for convergence in auto-calibration (default is 1e-10).
        verbose (bool): Whether to print detailed information during preprocessing (default is False).

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing columns 'X', 'Y', 'Z', and 'wear'.
    """

    _df = df.copy()

    epoch_size = preprocess_args.get('autocalib_epoch_size', 10)
    max_iter = preprocess_args.get('autocalib_max_iter', 1000)
    tol = preprocess_args.get('autocalib_tol', 1e-10)
    sd_criter = preprocess_args.get('autocalib_sd_criter', 0.013)
    mean_criter = preprocess_args.get('autocalib_mean_criter', 2)
    sphere_crit = preprocess_args.get('autocalib_sphere_crit', 0.3)

    _df = auto_calibrate(_df, sf, meta_dict, epoch_size, max_iter, tol, sd_criter, mean_criter, sphere_crit, verbose=verbose)
    if verbose:
        print('Calibration done')

    filter_type = preprocess_args.get('filter_type', 'highpass')
    filter_cutoff = preprocess_args.get('filter_cutoff', 15)

    if (filter_type == 'bandpass' or filter_type == 'bandstop') and (type(filter_cutoff) != list or len(filter_cutoff) != 2):
        raise ValueError('Bandpass and bandstop filters require a list of two cutoff frequencies.')

    if (filter_type == 'highpass' or filter_type == 'lowpass') and type(filter_cutoff) not in [float, int]:
        raise ValueError('Highpass and lowpass filters require a single cutoff frequency.')

    _df = remove_noise(_df, sf, filter_type, filter_cutoff)
    if verbose:
        print('Noise removal done')

    std_threshold = preprocess_args.get('wear_detection_std_threshold', 0.013)
    range_threshold = preprocess_args.get('wear_detection_range_threshold', 0.15)

    _df['wear'] = detect_wear(_df, sf, std_threshold, range_threshold)['wear']
    if verbose:
        print('Wear detection done')

    total, wear, nonwear = calc_weartime(_df, sf)
    meta_dict.update({'resampled_total_time': total, 'resampled_wear_time': wear, 'resampled_non-wear_time': nonwear})
    if verbose:
        print('Wear time calculated')

    return _df[['X', 'Y', 'Z', 'wear']]


def calibrate():
    pass

def remove_noise(df: pd.DataFrame, sf: float, filter_type: str = 'lowpass', filter_cutoff: float = 2) -> pd.DataFrame:
    """
    Remove noise from accelerometer data using a Butterworth low-pass filter.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        cutoff (float): Cutoff frequency for the low-pass filter in Hz (default is 2.5).
        fs (float): Sampling frequency of the accelerometer data in Hz (default is 50).
        order (int): Order of the Butterworth filter (default is 2).

    Returns:
        pd.DataFrame: DataFrame with noise removed from the 'X', 'Y', and 'Z' columns.
    """

    if df.empty:
        raise ValueError("Dataframe is empty.")

    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        raise KeyError("Dataframe must contain 'X', 'Y' and 'Z' columns.")

    def butter_lowpass_filter(data, cutoff, sf, btype, order=2):
        # Design Butterworth filter
        nyquist = 0.5 * sf  # Nyquist frequency
        normal_cutoff = np.array(cutoff) / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)

        # Apply filter to data
        return filtfilt(b, a, data)

    _df = df.copy()

    cutoff = filter_cutoff
    _df['X'] = butter_lowpass_filter(_df['X'], cutoff, sf, btype=filter_type)
    _df['Y'] = butter_lowpass_filter(_df['Y'], cutoff, sf, btype=filter_type)
    _df['Z'] = butter_lowpass_filter(_df['Z'], cutoff, sf, btype=filter_type)

    return _df

def detect_non_wear():
    pass

def calc_weartime(df: pd.DataFrame, sf: float) -> Tuple[float, float, float]:
    """
    Calculate total, wear, and non-wear time from accelerometer data.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with a 'wear' column.
        sf (float): Sampling frequency of the accelerometer data in Hz.

    Returns:
        Tuple[float, float, float]: A tuple containing total time, wear time, and non-wear time in seconds.
    """

    total = float((df.index[-1] - df.index[0]).total_seconds())
    wear = float((df['wear'].sum()) * (1 / sf))
    nonwear = float((total - wear))

    return total, wear, nonwear



    """
    Calculate the rolling standard deviation of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing data to calculate the rolling standard deviation.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Array containing the rolling standard deviation values.
    """

    if len(df) < window_size:
        raise ValueError("Window size is larger than the number of data points.")

    if window_size <= 0:
        raise ValueError("Window size must be greater than 0.")

    if df.size == 0:
        raise ValueError("Dataframe is empty.")

    return pd.Series(df).rolling(window=window_size, min_periods=1).std().fillna(0).values

