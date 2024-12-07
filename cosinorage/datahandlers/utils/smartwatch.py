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

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from typing import Tuple, Optional, Dict
from glob import glob
from skdh.preprocessing import CountWearDetection, CalibrateAccelerometer


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


def preprocess_smartwatch_data(data: pd.DataFrame, sf: float, meta_dict: dict, preprocess_args: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Preprocess smartwatch data by performing auto-calibration, noise removal, and wear detection.

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        sf (float): Sampling frequency of the accelerometer data in Hz.
        meta_dict (dict): Dictionary to store metadata about the preprocessing steps.
        preprocess_args (dict): Dictionary containing preprocessing parameters:
            - autocalib_sphere_crit (float): Sphere criterion for auto-calibration (default: 1)
            - autocalib_sd_criter (float): Standard deviation criterion for auto-calibration (default: 0.3)
            - filter_type (str): Type of filter to use ('highpass', 'lowpass', 'bandpass', 'bandstop')
            - filter_cutoff (float or list): Cutoff frequency/frequencies for the filter
        verbose (bool): Whether to print detailed information during preprocessing.

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing columns 'X', 'Y', 'Z', and 'wear'.
    """

    _data = data.copy()

    # calibration
    sphere_crit = preprocess_args.get('autocalib_sphere_crit', 1)
    sd_criter = preprocess_args.get('autocalib_sd_criter', 0.3)
    _data = calibrate(data, sf=sf, sphere_crit=sphere_crit, sd_criteria=sd_criter, meta_dict=meta_dict, verbose=verbose)

    # noise removal
    type = preprocess_args.get('filter_type', 'highpass')
    cutoff = preprocess_args.get('filter_cutoff', 15)
    _data = remove_noise(_data, sf=sf, filter_type=type, filter_cutoff=cutoff, verbose=verbose)

    # wear detection
    _data['wear'] = detect_wear(_data, sf, meta_dict=meta_dict, verbose=verbose)

    total, wear, nonwear = calc_weartime(_data, sf=sf, meta_dict=meta_dict, verbose=verbose)

    return _data[['X', 'Y', 'Z', 'wear']]


def calibrate(data: pd.DataFrame, sf: float, sphere_crit: float, sd_criteria: float, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Calibrate accelerometer data using auto-calibration techniques.

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        sf (float): Sampling frequency of the accelerometer data in Hz.
        sphere_crit (float): Sphere criterion for auto-calibration.
        sd_criteria (float): Standard deviation criterion for auto-calibration.
        meta_dict (dict): Dictionary to store calibration metadata.
        verbose (bool): Whether to print detailed information during calibration.

    Returns:
        pd.DataFrame: Calibrated accelerometer data with columns 'X', 'Y', and 'Z'.
    """
    _data = data.copy()

    time = np.array(_data.index.astype('int64') // 10 ** 9)
    acc = np.array(_data[["X", "Y", "Z"]]).astype(np.float64) / 1000

    calibrator = CalibrateAccelerometer(sphere_crit=sphere_crit, sd_criteria=sd_criteria)
    result = calibrator.predict(time=time, accel=acc, fs=sf)

    # If no calibration was performed, result will be None or won't contain 'accel'
    if result is None or 'accel' not in result:
        # Return the original data, converted to g units
        _data = pd.DataFrame(acc, columns=['X', 'Y', 'Z'])
    else:
        _data = pd.DataFrame(result['accel'], columns=['X', 'Y', 'Z'])
    
    _data.set_index(data.index, inplace=True)

    if result is not None:
        meta_dict.update({
            'calibration_offset': result.get('offset', None),
            'calibration_scale': result.get('scale', None)
        })

    if verbose:
        print('Calibration done')

    return _data


def remove_noise(df: pd.DataFrame, sf: float, filter_type: str = 'lowpass', filter_cutoff: float = 2, verbose: bool = False) -> pd.DataFrame:
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
    if (filter_type == 'bandpass' or filter_type == 'bandstop') and (type(filter_cutoff) != list or len(filter_cutoff) != 2):
        raise ValueError('Bandpass and bandstop filters require a list of two cutoff frequencies.')

    if (filter_type == 'highpass' or filter_type == 'lowpass') and type(filter_cutoff) not in [float, int]:
        raise ValueError('Highpass and lowpass filters require a single cutoff frequency.')

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

    if verbose:
        print('Noise removal done')

    return _df


def detect_wear(data: pd.DataFrame, sf: float, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Detect periods of device wear using count-based wear detection.

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with columns 'X', 'Y', and 'Z'.
        sf (float): Sampling frequency of the accelerometer data in Hz.
        meta_dict (dict): Dictionary to store wear detection metadata.
        verbose (bool): Whether to print detailed information during wear detection.

    Returns:
        pd.DataFrame: DataFrame containing a 'wear' column with binary wear status (1 for wear, 0 for non-wear).
    """
    _data = data.copy()

    time = np.array(_data.index.astype('int64') // 10 ** 9)
    acc = np.array(_data[["X", "Y", "Z"]]).astype(np.float64) / 1000

    wear_predictor = CountWearDetection()
    ranges = wear_predictor.predict(time=time, accel=acc, fs=sf)['wear']

    wear_array = np.zeros(len(data.index))
    for start, end in ranges:
        wear_array[start:end + 1] = 1

    _data['wear'] = pd.DataFrame(wear_array, columns=['wear']).set_index(data.index)

    if verbose:
        print('Wear detection done')

    return _data[['wear']]


def calc_weartime(data: pd.DataFrame, sf: float, meta_dict: dict, verbose: bool) -> Tuple[float, float, float]:
    """
    Calculate total, wear, and non-wear time from accelerometer data.

    This function analyzes the wear detection data to compute the total recording duration,
    the time the device was worn, and the time it wasn't worn. The results are stored in
    the provided metadata dictionary and returned as a tuple.

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with a 'wear' column
            where 1 indicates wear and 0 indicates non-wear.
        sf (float): Sampling frequency of the accelerometer data in Hz.
        meta_dict (dict): Dictionary to store the calculated wear times under the keys
            'resampled_total_time', 'resampled_wear_time', and 'resampled_non-wear_time'.
        verbose (bool): If True, prints a confirmation message when wear time calculation
            is complete.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - total_time (float): Total recording duration in seconds
            - wear_time (float): Time the device was worn in seconds
            - non_wear_time (float): Time the device wasn't worn in seconds

    Notes:
        - The wear time is calculated by summing the 'wear' column and dividing by the
          sampling frequency to convert from samples to seconds
        - The total time is calculated as the difference between the first and last
          timestamp in the index
        - The non-wear time is calculated as the difference between total time and wear time
    """
    _data = data.copy()

    total = float((_data.index[-1] - _data.index[0]).total_seconds())
    wear = float((_data['wear'].sum()) * (1 / sf))
    nonwear = float((total - wear))

    meta_dict.update({'resampled_total_time': total, 'resampled_wear_time': wear, 'resampled_non-wear_time': nonwear})
    if verbose:
        print('Wear time calculated')

    return total, wear, nonwear



