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
import os
import numpy as np
from typing import Tuple
from skdh.preprocessing import CalibrateAccelerometer, AccelThresholdWearDetection
from scipy.signal import butter, filtfilt
from claid.data_collection.load.load_sensor_data import *

from .filtering import filter_incomplete_days, filter_consecutive_days
from .calc_enmo import calculate_enmo


def read_galaxy_data(galaxy_file_dir: str, meta_dict: dict, verbose: bool = False):
    """
    Read accelerometer data from Galaxy Watch binary files.

    Args:
        galaxy_file_dir (str): Directory containing Galaxy Watch data files
        meta_dict (dict): Dictionary to store metadata about the loaded data
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: DataFrame containing accelerometer data with columns ['X', 'Y', 'Z']
    """

    data = pd.DataFrame()

    n_files = 0
    for day_dir in os.listdir(galaxy_file_dir):
        if os.path.isdir(galaxy_file_dir + day_dir):
            for file in os.listdir(galaxy_file_dir + day_dir):
                # only consider binary files
                if file.endswith(".binary") and file.startswith("acceleration_data"):
                    _temp = acceleration_data_to_dataframe(load_acceleration_data(galaxy_file_dir + day_dir + "/" + file))
                    data = pd.concat([data, _temp])
                    n_files += 1

    if verbose:
        print(f"Read {n_files} files from {galaxy_file_dir}")

    data = data.rename(columns={'unix_timestamp_in_ms': 'TIMESTAMP', 'acceleration_x': 'X', 'acceleration_y': 'Y', 'acceleration_z': 'Z'})
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms')
    data.set_index('TIMESTAMP', inplace=True)
    data.drop(columns=['effective_time_frame', 'sensor_body_location'], inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(f"Loaded {data.shape[0]} accelerometer data records from {galaxy_file_dir}")

    meta_dict['raw_n_datapoints'] = data.shape[0]
    meta_dict['raw_start_datetime'] = data.index.min()
    meta_dict['raw_end_datetime'] = data.index.max()
    meta_dict['raw_data_frequency'] = '25Hz'
    meta_dict['raw_data_type'] = 'accelerometer'
    meta_dict['raw_data_unit'] = 'custom'

    return data


def filter_galaxy_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False, preprocess_args: dict = {}) -> pd.DataFrame:
    """
    Filter Galaxy Watch accelerometer data by removing incomplete days and selecting longest consecutive sequence.

    Args:
        data (pd.DataFrame): Raw accelerometer data
        meta_dict (dict): Dictionary to store metadata about the filtering process
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Filtered accelerometer data
    """
    _data = data.copy()

    # filter out first and last day
    n_old = _data.shape[0]
    _data = _data.loc[(_data.index.date != _data.index.date.min()) & (_data.index.date != _data.index.date.max())]
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{_data.shape[0]} accelerometer records due to filtering out first and last day")

    # filter out sparse days
    required_points_per_day = preprocess_args.get('required_daily_coverage', 0.5) * 2160000
    n_old = _data.shape[0]
    _data = filter_incomplete_days(_data, data_freq=25, expected_points_per_day=required_points_per_day)
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{n_old} accelerometer records due to incomplete daily coverage")

    # filter for longest consecutive sequence of days
    old_n = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(f"Filtered out {old_n - _data.shape[0]}/{old_n} minute-level accelerometer records due to filtering for longest consecutive sequence of days")

    return _data


def resample_galaxy_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Resample Galaxy Watch accelerometer data to a regular 40ms interval (25Hz).

    Args:
        data (pd.DataFrame): Filtered accelerometer data
        meta_dict (dict): Dictionary to store metadata about the resampling process
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Resampled accelerometer data at 25Hz
    """
    _data = data.copy()

    n_old = _data.shape[0]
    _data = _data.resample('40ms').interpolate(method='linear').bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    return _data


def preprocess_galaxy_data(data: pd.DataFrame, preprocess_args: dict = {}, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Preprocess Galaxy Watch accelerometer data including rescaling, calibration, noise removal, and wear detection.

    Args:
        data (pd.DataFrame): Resampled accelerometer data
        preprocess_args (dict): Dictionary containing preprocessing parameters
        meta_dict (dict): Dictionary to store metadata about the preprocessing
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Preprocessed accelerometer data with additional columns for raw values and wear detection
    """
    _data = data.copy()
    _data[['X_raw', 'Y_raw', 'Z_raw']] = _data[['X', 'Y', 'Z']]

    # recaling of accelerometer data according to blog post: https://developer.samsung.com/sdp/blog/en/2025/04/10/understanding-and-converting-galaxy-watch-accelerometer-data
    _data[['X', 'Y', 'Z']] = _data[['X', 'Y', 'Z']] / 4096

    # calibration
    sphere_crit = preprocess_args.get('autocalib_sphere_crit', 1)
    sd_criter = preprocess_args.get('autocalib_sd_criter', 0.3)
    _data[['X', 'Y', 'Z']] = calibrate(_data, sf=25, sphere_crit=sphere_crit, sd_criteria=sd_criter, meta_dict=meta_dict, verbose=verbose)

    # noise removal
    type = preprocess_args.get('filter_type', 'highpass')
    cutoff = preprocess_args.get('filter_cutoff', 15)
    _data[['X', 'Y', 'Z']] = remove_noise(_data, sf=25, filter_type=type, filter_cutoff=cutoff, verbose=verbose)

    # wear detection
    sd_crit = preprocess_args.get('wear_sd_crit', 0.00013)
    range_crit = preprocess_args.get('wear_range_crit', 0.00067)
    window_length = preprocess_args.get('wear_window_length', 30)
    window_skip = preprocess_args.get('wear_window_skip', 7)
    _data['wear'] = detect_wear(_data, 25, sd_crit, range_crit, window_length, window_skip, meta_dict=meta_dict, verbose=verbose)

    # calculate total, wear, and non-wear time
    calc_weartime(_data, sf=25, meta_dict=meta_dict, verbose=verbose)

    _data['ENMO'] = calculate_enmo(_data, verbose=verbose) * 1000

    if verbose:
        print(f"Preprocessed accelerometer data")

    return _data


def acceleration_data_to_dataframe(data):
    """
    Convert binary acceleration data to pandas DataFrame.

    Args:
        data: Binary acceleration data object

    Returns:
        pd.DataFrame: DataFrame containing accelerometer data with timestamps and sensor information
    """
    rows = []
    for sample in data.samples:
        rows.append({
            'acceleration_x': sample.acceleration_x,
            'acceleration_y': sample.acceleration_y,
            'acceleration_z': sample.acceleration_z,
            'sensor_body_location': sample.sensor_body_location,
            'unix_timestamp_in_ms': sample.unix_timestamp_in_ms,
            'effective_time_frame': sample.effective_time_frame
        })

    return pd.DataFrame(rows)


def calibrate(data: pd.DataFrame, sf: float, sphere_crit: float, sd_criteria: float, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Calibrate accelerometer data using sphere fitting method.

    Args:
        data (pd.DataFrame): Raw accelerometer data
        sf (float): Sampling frequency in Hz
        sphere_crit (float): Sphere fitting criterion threshold
        sd_criteria (float): Standard deviation criterion threshold
        meta_dict (dict): Dictionary to store calibration parameters
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Calibrated accelerometer data
    """

    _data = data.copy()

    time = np.array(_data.index.astype('int64') // 10 ** 9)
    acc = np.array(_data[["X", "Y", "Z"]]).astype(np.float64)

    calibrator = CalibrateAccelerometer(sphere_crit=sphere_crit, sd_criteria=sd_criteria)
    result = calibrator.predict(time=time, accel=acc, fs=sf)

    _data = pd.DataFrame(result['accel'], columns=['X', 'Y', 'Z'])
    _data.set_index(data.index, inplace=True)

    meta_dict.update({'calibration_offset': result['offset']})
    meta_dict.update({'calibration_scale': result['scale']})

    if verbose:
        print('Calibration done')

    return _data[['X', 'Y', 'Z']]


def remove_noise(data: pd.DataFrame, sf: float, filter_type: str = 'lowpass', filter_cutoff: float = 2, verbose: bool = False) -> pd.DataFrame:
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

    if data.empty:
        raise ValueError("Dataframe is empty.")

    if not all(col in data.columns for col in ['X', 'Y', 'Z']):
        raise KeyError("Dataframe must contain 'X', 'Y' and 'Z' columns.")

    def butter_lowpass_filter(data, cutoff, sf, btype, order=2):
        # Design Butterworth filter
        nyquist = 0.5 * sf  # Nyquist frequency
        normal_cutoff = np.array(cutoff) / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)

        # Apply filter to data
        return filtfilt(b, a, data)

    _data = data.copy()

    cutoff = filter_cutoff
    _data['X'] = butter_lowpass_filter(_data['X'], cutoff, sf, btype=filter_type)
    _data['Y'] = butter_lowpass_filter(_data['Y'], cutoff, sf, btype=filter_type)
    _data['Z'] = butter_lowpass_filter(_data['Z'], cutoff, sf, btype=filter_type)

    if verbose:
        print('Noise removal done')

    return _data[['X', 'Y', 'Z']]


def detect_wear(data: pd.DataFrame, sf: float, sd_crit: float, range_crit: float, window_length: int, window_skip: int, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Detect periods of device wear using acceleration thresholds.

    Args:
        data (pd.DataFrame): Preprocessed accelerometer data
        sf (float): Sampling frequency in Hz
        sd_crit (float): Standard deviation criterion for wear detection
        range_crit (float): Range criterion for wear detection
        window_length (int): Length of sliding window in seconds
        window_skip (int): Number of seconds to skip between windows
        meta_dict (dict): Dictionary to store wear detection metadata
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: DataFrame with binary wear detection column
    """
    _data = data.copy()

    time = np.array(_data.index.astype('int64') // 10 ** 9)
    acc = np.array(_data[["X", "Y", "Z"]]).astype(np.float64) / 1000

    #wear_predictor = CountWearDetection()
    wear_predictor = AccelThresholdWearDetection(sd_crit=sd_crit, range_crit=range_crit, window_length=window_length, window_skip=window_skip)
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

    Args:
        data (pd.DataFrame): DataFrame containing accelerometer data with a 'wear' column
        sf (float): Sampling frequency of the accelerometer data in Hz
        meta_dict (dict): Dictionary to store wear time metadata
        verbose (bool): Whether to print progress information

    Returns:
        None: Updates meta_dict with the following keys:
            - resampled_total_time: Total recording time in seconds
            - resampled_wear_time: Time device was worn in seconds
            - resampled_non-wear_time: Time device was not worn in seconds
    """
    _data = data.copy()

    total = float((_data.index[-1] - _data.index[0]).total_seconds())
    wear = float((_data['wear'].sum()) * (1 / sf))
    nonwear = float((total - wear))

    meta_dict.update({'total_time': total, 'wear_time': wear, 'non-wear_time': nonwear})
    if verbose:
        print('Wear time calculated')