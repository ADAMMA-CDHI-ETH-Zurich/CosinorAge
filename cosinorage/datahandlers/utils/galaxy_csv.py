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
from claid.data_collection.load.load_sensor_data import *

from .filtering import filter_incomplete_days, filter_consecutive_days

def read_galaxy_csv_data(galaxy_file_path: str, meta_dict: dict, verbose: bool = False):
    """
    Read ENMO data from Galaxy Watch csv file.

    Args:
        galaxy_file_path (str): Path to the Galaxy Watch data file
        meta_dict (dict): Dictionary to store metadata about the loaded data
        verbose (bool): Whether to print progress information
    Returns:
        pd.DataFrame: DataFrame containing ENMO data with columns ['TIMESTAMP', 'ENMO']
    """

    data = pd.read_csv(galaxy_file_path)

    if verbose:
        print(f"Read csv file from {galaxy_file_path}")

    data = data.rename(columns={'time': 'TIMESTAMP', 'enmo_mg': 'ENMO'})
    # Convert UTC timestamps to local time
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP']).dt.tz_localize(None)
    data.set_index('TIMESTAMP', inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(f"Loaded {data.shape[0]} ENMO data records from {galaxy_file_path}")

    meta_dict['raw_n_datapoints'] = data.shape[0]
    meta_dict['raw_start_datetime'] = data.index.min()
    meta_dict['raw_end_datetime'] = data.index.max()
    meta_dict['raw_data_frequency'] = '1/60Hz'
    meta_dict['raw_data_type'] = 'ENMO'
    meta_dict['raw_data_unit'] = 'mg'

    return data


def filter_galaxy_csv_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False, preprocess_args: dict = {}) -> pd.DataFrame:
    """
    Filter Galaxy Watch ENMO data by removing incomplete days and selecting longest consecutive sequence.

    Args:
        data (pd.DataFrame): Raw ENMO data
        meta_dict (dict): Dictionary to store metadata about the filtering process
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Filtered ENMO data
    """
    _data = data.copy()

    # filter out sparse days
    required_points_per_day = preprocess_args.get('required_daily_coverage', 0.5) * 1440
    n_old = _data.shape[0]
    _data = filter_incomplete_days(_data, data_freq=1/60, expected_points_per_day=required_points_per_day)
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{n_old} ENMO records due to incomplete daily coverage")

    # filter for longest consecutive sequence of days
    n_old = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{n_old} ENMO records due to filtering for longest consecutive sequence of days")

    # resample to minute-level
    _data = _data.resample('1min').interpolate(method='linear').bfill()
    n_old = _data.shape[0]
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    # filter out first and last day if it is incomplete (not 1440 samples)
    n_old = _data.shape[0]
    
    # Get the first and last day
    first_day = _data.index[0].date()
    last_day = _data.index[-1].date()
    
    # Filter out first day if incomplete
    if len(_data[_data.index.date == first_day]) != 1440:
        _data = _data[_data.index.date > first_day]
    
    # Filter out last day if incomplete
    if len(_data[_data.index.date == last_day]) != 1440:
        _data = _data[_data.index.date < last_day]
    
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{n_old} ENMO records due to filtering out first and last day")

    return _data


def resample_galaxy_csv_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Ensure we have minute-level data across the whole timeseries. 

    Args:
        data (pd.DataFrame): Filtered ENMO data
        meta_dict (dict): Dictionary to store metadata about the resampling process
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Resampled ENMO data to minute-level.
    """
    _data = data.copy()

    n_old = _data.shape[0]
    _data = _data.resample('1min').interpolate(method='linear').bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    return _data


def preprocess_galaxy_csv_data(data: pd.DataFrame, preprocess_args: dict = {}, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Preprocess Galaxy Watch ENMO data including rescaling, calibration, noise removal, and wear detection.

    Args:
        data (pd.DataFrame): Resampled ENMO data
        preprocess_args (dict): Dictionary containing preprocessing parameters
        meta_dict (dict): Dictionary to store metadata about the preprocessing
        verbose (bool): Whether to print progress information

    Returns:
        pd.DataFrame: Preprocessed ENMO data with additional columns for raw values and wear detection
    """
    _data = data.copy()

    # wear detection - not implemented for enmo data yet (current algorithm relies on accelerometer data)
    _data['wear'] = -1

    if verbose:
        print(f"Preprocessed ENMO data")

    return _data