import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Tuple, Optional
from glob import glob


def read_smartwatch_data(directory_path: str) -> Tuple[pd.DataFrame, Optional[float]]:
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


def preprocess_smartwatch_data(df: pd.DataFrame, sf: float, meta_dict: dict, epoch_size: int = 10, max_iter: int = 1000,
                               tol: float = 1e-10, verbose: bool = False) -> pd.DataFrame:
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

    _df = auto_calibrate(_df, sf, epoch_size, max_iter, tol)
    if verbose:
        print('Calibration done')

    _df = remove_noise(_df)
    if verbose:
        print('Noise removal done')

    _df['wear'] = detect_wear(_df, sf)['wear']
    if verbose:
        print('Wear detection done')

    total, wear, nonwear = calc_weartime(_df, sf)
    meta_dict.update({'total time': total, 'wear time': wear, 'non-wear time': nonwear})
    if verbose:
        print('Wear time calculated')

    return _df[['X', 'Y', 'Z', 'wear']]


def auto_calibrate(df: pd.DataFrame, sf: float, epoch_size: int = 10, max_iter: int = 1000,
                   tol: float = 1e-10) -> pd.DataFrame:
    """
    Perform autocalibration on accelerometer data, adjusting offset and scale to reduce calibration error. The implementation is based on the algorithm described in the GGIR R-package.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        sf (float): Sampling frequency of the accelerometer data in Hz.
        epoch_size (int): Epoch size for calibration in seconds (default is 10).
        max_iter (int): Maximum number of iterations for auto-calibration (default is 1000).
        tol (float): Tolerance for convergence in auto-calibration (default is 1e-10).

    Returns:
        pd.DataFrame: Calibrated DataFrame with adjusted offset and scale.
    """
    # Initialize calibration parameters for scale and offset
    scale = np.array([1.0, 1.0, 1.0])  # Start with no scaling (1.0) for x, y, z
    offset = np.array([0.0, 0.0, 0.0])  # Start with no offset for x, y, z
    calib_error_start, calib_error_end = None, None  # Track calibration error before and after adjustment

    # Step 1: Calculate features for calibration
    gx, gy, gz = df['X'].values, df['Y'].values, df['Z'].values
    en = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # Roll mean and standard deviation
    window_size = int(sf * epoch_size)
    mean_en = _roll_mean(en, window_size)
    mean_gx = _roll_mean(gx, window_size)
    mean_gy = _roll_mean(gy, window_size)
    mean_gz = _roll_mean(gz, window_size)

    sd_gx = _roll_sd(gx, window_size)
    sd_gy = _roll_sd(gy, window_size)
    sd_gz = _roll_sd(gz, window_size)

    # Step 2: Filter features for nonmovement periods based on low standard deviation
    sd_criter = 0.013  # Example threshold for standard deviation

    nonmovement_idx = np.where(
        (sd_gx < sd_criter) & (sd_gy < sd_criter) & (sd_gz < sd_criter) & (mean_gx < 2) & (mean_gy < 2) & (
                mean_gz < 2))[0]

    mean_gx = mean_gx[nonmovement_idx]
    mean_gy = mean_gy[nonmovement_idx]
    mean_gz = mean_gz[nonmovement_idx]

    # Step 3: Ensure enough data points for calibration
    if len(mean_gx) > 10:
        # Calculate initial calibration error based on distance from expected 1g magnitude
        calib_error_start = np.mean(
            np.abs(np.sqrt(mean_gx ** 2 + mean_gy ** 2 + mean_gz ** 2) - 1))

        # Step 4: Iterative adjustment to minimize calibration error

        input_data = np.vstack((mean_gx, mean_gy, mean_gz)).T
        weights = np.ones(len(input_data))  # Initialize weights for each data point
        res = float('inf')

        # Iterative loop to adjust scale and offset
        for iteration in tqdm(range(max_iter), desc="Calibrating", unit="iter"):
            # Apply current offset and scale to data
            adjusted = (input_data - offset) * scale
            norms = np.linalg.norm(adjusted, axis=1, keepdims=True)  # Compute norms for each row
            closest_point = adjusted / norms  # Normalize to project onto the unit sphere
            # Offset and scale changes for each axis
            offset_change = np.zeros(3)
            scale_change = np.ones(3)
            # Adjust offset and scale for each axis using linear regression
            for k in range(3):
                model = LinearRegression()
                model.fit(adjusted[:, k].reshape(-1, 1), closest_point[:, k], sample_weight=weights)
                offset_change[k] = model.intercept_  # Intercept represents offset adjustment
                scale_change[k] = model.coef_[0]  # Coefficient represents scale adjustment
                adjusted[:, k] = model.predict(adjusted[:, k].reshape(-1, 1))  # Update adjusted data
            # Apply changes to offset and scale
            offset += offset_change / (scale * scale_change)
            scale *= scale_change
            # Check convergence based on change in residuals
            new_res = np.mean(weights * np.sum((adjusted - closest_point) ** 2, axis=1))
            weights = np.minimum(1 / np.sqrt(np.sum((adjusted - closest_point) ** 2, axis=1)), 1 / 0.01)
            if abs(new_res - res) < tol:
                tqdm.write(f"Convergence reached at iteration {iteration + 1}")
                break  # Stop iteration if convergence criterion met
            res = new_res  # Update residual for next iteration
        # Calculate final calibration error
        calib_error_end = np.mean(
            np.abs(np.sqrt(mean_gx ** 2 + mean_gy ** 2 + mean_gz ** 2) - 1))
        # Output summary of calibration results if verbose is enabled
    else:
        print("Insufficient nonmovement data for calibration.")
    # Return calibration results
    return offset + df * scale


def remove_noise(df: pd.DataFrame, cutoff: float = 2.5, fs: float = 50, order: int = 2) -> pd.DataFrame:
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

    def butter_lowpass_filter(data, cutoff, fs, order=2):
        # Design Butterworth filter
        nyquist = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Apply filter to data
        return filtfilt(b, a, data)

    df['X'] = butter_lowpass_filter(df['X'], cutoff, fs)
    df['Y'] = butter_lowpass_filter(df['Y'], cutoff, fs)
    df['Z'] = butter_lowpass_filter(df['Z'], cutoff, fs)

    return df


def detect_wear(df: pd.DataFrame, sf: float) -> pd.DataFrame:
    """
    Detect wear and non-wear periods from accelerometer data. The implementation is based on the algorithm described in the BioPsyKit package.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        sf (float): Sampling frequency of the accelerometer data in Hz.

    Returns:
        pd.DataFrame: DataFrame with a 'wear' column indicating wear (1) and non-wear (0) periods.
    """

    # copy and rename acc columns of the dataframe
    _df = df[['X', 'Y', 'Z']].copy()

    wnw = _detect_wear(_df, sf)

    # bring wear detection results back to the original frequency
    start_time = wnw['start'].min()
    end_time = wnw['end'].max()
    interval_ms = 1000 / sf
    new_index = pd.date_range(start=start_time, end=end_time, freq=f'{interval_ms}ms')

    # Expand the DataFrame to cover each timestamp in the range of 'start' to 'end' for each row
    expanded_data = []
    for _, row in wnw.iterrows():
        time_range = pd.date_range(start=row['start'], end=row['end'], freq=f'{interval_ms}ms')
        wear_values = pd.Series(row['wear'], index=time_range)
        expanded_data.append(wear_values)

    # Concatenate all the expanded wear values into a single Series and reindex to the full range
    wear_series = pd.concat(expanded_data)

    # Calculate the mean for each time period, which averages any overlaps
    wear_mean = wear_series.groupby(level=0).mean().to_frame(name='wear')
    return wear_mean


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
    nonwear = float((total - wear) * (1 / sf))

    return total, wear, nonwear


def _roll_mean(df, window_size):
    """
    Calculate the rolling mean of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing data to calculate the rolling mean.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Array containing the rolling mean values.
    """

    return np.convolve(df, np.ones(window_size) / window_size, mode='valid')


def _roll_sd(df, window_size):
    """
    Calculate the rolling standard deviation of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing data to calculate the rolling standard deviation.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Array containing the rolling standard deviation values.
    """

    return pd.Series(df).rolling(window=window_size).std().dropna().values


def _detect_wear(data: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    """
    Detect non-wear times from raw acceleration data. The implementation is based on the algorithm described in the BioPsyKit package.

    Args:
        data (pd.DataFrame): Input acceleration data with columns for each axis ('X', 'Y', 'Z').
        sampling_rate (float): Sampling rate of the recorded data in Hz.

    Returns:
        pd.DataFrame: DataFrame with wear (1) and non-wear (0) times per 15-minute interval.
    """
    # Ensure data contains acceleration columns
    if not all(axis in data.columns for axis in ['X', 'Y', 'Z']):
        raise ValueError("Data must contain 'X', 'Y' and 'Z' columns.")

    index = data.index if isinstance(data.index, pd.DatetimeIndex) else None

    # Parameters
    window = 20  # Window size in minutes should be 60
    overlap = 5  # Overlap size in minutes should be 15
    overlap_percent = 1.0 - (overlap / window)

    window_samples = int(window * 60 * sampling_rate)
    step_samples = int(window_samples * (1 - overlap_percent))

    # Apply sliding window to each axis
    acc_sliding = {
        col: _sliding_window(data[col].values, window_samples, step_samples)
        for col in ['X', 'Y', 'Z']
    }

    # Resample index if available
    if index is not None:
        index_resample = _resample_index(index, window_samples, step_samples)
    else:
        index_resample = None

    # Calculate standard deviation and range for each window
    acc_std = pd.DataFrame({
        axis: np.nanstd(acc_sliding[axis], ddof=1, axis=1)
        for axis in acc_sliding
    })

    acc_range = pd.DataFrame({
        axis: np.nanmax(acc_sliding[axis], axis=1) - np.nanmin(acc_sliding[axis], axis=1)
        for axis in acc_sliding
    })

    # Apply thresholds
    acc_std_threshold = 0.013
    acc_range_threshold = 0.15

    acc_std_binary = (acc_std >= acc_std_threshold).astype(int)
    acc_range_binary = (acc_range >= acc_range_threshold).astype(int)

    # Sum across axes
    acc_std_sum = acc_std_binary.sum(axis=1)
    acc_range_sum = acc_range_binary.sum(axis=1)

    # Initial wear detection
    wear = np.ones_like(acc_std_sum)
    wear[(acc_std_sum < 1) | (acc_range_sum < 1)] = 0  # Non-wear periods

    wear = pd.DataFrame({'wear': wear})
    if index_resample is not None:
        wear = wear.join(index_resample)

    # Rescore wear detection
    for _ in range(3):
        wear = _rescore_wear_detection(wear)

    return wear


def _resample_index(index, window_samples, step_samples):
    """
    Resample the index to match the window and step sizes.

    Args:
        index (pd.Index): Original index to be resampled.
        window_samples (int): Number of samples in each window.
        step_samples (int): Number of samples to step between windows.

    Returns:
        pd.DataFrame: DataFrame with 'start' and 'end' columns indicating the start and end times of each window.
    """

    indices = np.arange(len(index))
    windows = _sliding_window(indices, window_samples, step_samples)
    start_end = windows[:, [0, -1]]

    if isinstance(index, pd.DatetimeIndex):
        start_times = index[start_end[:, 0]]
        end_times = index[start_end[:, 1]]
        index_resample = pd.DataFrame({'start': start_times, 'end': end_times})
    else:
        index_resample = pd.DataFrame({'start': start_end[:, 0], 'end': start_end[:, 1]})

    return index_resample


def _rescore_wear_detection(wear_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rescore wear detection based on duration and surrounding blocks.

    Args:
        wear_data (pd.DataFrame): DataFrame with 'wear' column indicating wear (1) and non-wear (0) periods.

    Returns:
        pd.DataFrame: Rescored DataFrame with updated 'wear' column.
    """

    # Group into wear and non-wear blocks
    wear_data = wear_data.copy()
    wear_data['block'] = (wear_data['wear'] != wear_data['wear'].shift()).cumsum()
    blocks = wear_data.groupby('block')

    # Convert duration from intervals to hours (each interval is 15 minutes)
    interval_duration = 0.25  # Hours

    # Iterate over blocks to rescore
    wear_list = []
    for i, group in enumerate(blocks):
        idx, block = group
        duration = len(block) * interval_duration
        wear_value = block['wear'].iloc[0]

        # Skip first and last block
        if i == 0 or i == len(blocks) - 1:
            wear_list.append(block)
            continue

        # Get previous and next blocks
        prev_block = blocks.get_group(idx - 1)
        next_block = blocks.get_group(idx + 1)
        dur_prev = len(prev_block) * interval_duration
        dur_next = len(next_block) * interval_duration

        # Rescoring rules
        total_dur = dur_prev + dur_next
        if wear_value == 1:
            ratio = duration / total_dur if total_dur > 0 else 0
            if (duration < 3 and ratio < 0.8) or (duration < 6 and ratio < 0.3):
                block['wear'] = 0  # Rescore to non-wear
        wear_list.append(block)

    wear_rescored = pd.concat(wear_list).sort_index()
    return wear_rescored.drop(columns='block')


def _sliding_window(arr, window_size, step_size):
    """
    Generate a sliding window view of the input array.

    Args:
        arr (np.ndarray): Input array to generate sliding windows from.
        window_size (int): Size of each window.
        step_size (int): Step size between windows.

    Returns:
        np.ndarray: Array of sliding windows.
    """

    num_windows = ((len(arr) - window_size) // step_size) + 1
    shape = (num_windows, window_size)
    strides = (arr.strides[0] * step_size, arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
