import pandas as pd
import os
import numpy as np

from claid.data_collection.load.load_sensor_data import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from .smartwatch import preprocess_smartwatch_data
from .filtering import filter_incomplete_days, filter_consecutive_days


def read_galaxy_data(gw_file_dir: str, meta_dict: dict, verbose: bool = False):

    data = pd.DataFrame()

    n_files = 0
    for day_dir in os.listdir(gw_file_dir):
        if os.path.isdir(gw_file_dir + day_dir):
            for file in os.listdir(gw_file_dir + day_dir):
                # only consider binary files
                if file.endswith(".binary") and file.startswith("acceleration_data"):
                    _temp = acceleration_data_to_dataframe(load_acceleration_data(gw_file_dir + day_dir + "/" + file))
                    data = pd.concat([data, _temp])
                    n_files += 1

    if verbose:
        print(f"Read {n_files} files from {gw_file_dir}")

    data = data.rename(columns={'unix_timestamp_in_ms': 'TIMESTAMP', 'acceleration_x': 'X', 'acceleration_y': 'Y', 'acceleration_z': 'Z'})
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms')
    data.set_index('TIMESTAMP', inplace=True)
    data.drop(columns=['effective_time_frame', 'sensor_body_location'], inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(f"Loaded {data.shape[0]} accelerometer data records from {gw_file_dir}")

    meta_dict['raw_n_timesteps'] = data.shape[0]
    meta_dict['raw_n_days'] = len(np.unique(data.index.date))
    meta_dict['raw_start_datetime'] = data.index.min()
    meta_dict['raw_end_datetime'] = data.index.max()
    meta_dict['raw_frequency'] = 'irregular (~25Hz)'
    meta_dict['raw_datatype'] = 'accelerometer'
    meta_dict['raw_unit'] = ''

    return data


def filter_galaxy_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()

    # filter out first and last day
    n_old = _data.shape[0]
    _data = _data.loc[(_data.index.date != _data.index.date.min()) & (_data.index.date != _data.index.date.max())]
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{_data.shape[0]} accelerometer records due to filtering out first and last day")

    # filter out sparse days
    n_old = _data.shape[0]
    _data = filter_incomplete_days(_data, data_freq=25, expected_points_per_day=2000000)
    if verbose:
        print(f"Filtered out {n_old - _data.shape[0]}/{n_old} accelerometer records due to incomplete daily coverage")

    # filter for longest consecutive sequence of days
    old_n = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(f"Filtered out {old_n - _data.shape[0]}/{old_n} minute-level accelerometer records due to filtering for longest consecutive sequence of days")

    return _data


def resample_galaxy_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()

    n_old = _data.shape[0]
    _data = _data.resample('40ms').interpolate(method='linear').bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    meta_dict['resampled_n_timestamps'] = _data.shape[0]
    meta_dict['resampled_n_days'] = len(np.unique(_data.index.date))
    meta_dict['resampled_start_datetime'] = _data.index.min()
    meta_dict['resampled_end_datetime'] = _data.index.max()
    meta_dict['resampled_frequency'] = '25Hz'
    meta_dict['resampled_datatype'] = 'accelerometer'
    meta_dict['resampled_unit'] = ''

    return _data


def preprocess_galaxy_data(data: pd.DataFrame, preprocess_args: dict = {}, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    _data = data.copy()

    _data[['X_raw', 'Y_raw', 'Z_raw']] = _data[['X', 'Y', 'Z']]
    _data[['X', 'Y', 'Z', 'wear']] = preprocess_smartwatch_data(_data[['X', 'Y', 'Z']], 25, meta_dict, preprocess_args=preprocess_args, verbose=verbose)
    if verbose:
        print(f"Preprocessed accelerometer data")

    return _data


def acceleration_data_to_dataframe(data):
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