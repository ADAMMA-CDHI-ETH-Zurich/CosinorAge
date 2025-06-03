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
from datetime import datetime, timedelta
from tqdm import tqdm

from .calc_enmo import calculate_enmo
from .filtering import filter_incomplete_days, filter_consecutive_days


def read_nhanes_data(file_dir: str, seqn: str = None, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Read and process NHANES accelerometer data files for a specific person.

    Args:
        file_dir (str): Directory containing NHANES data files (PAXDAY, PAXHD, PAXMIN)
        seqn (str, optional): Unique identifier for the participant. Required.
        meta_dict (dict, optional): Dictionary to store metadata. Defaults to {}.
        verbose (bool, optional): Whether to print processing status. Defaults to False.

    Returns:
        pd.DataFrame: Processed accelerometer data with columns for X, Y, Z, wear, sleep, paxpredm, and ENMO,
                     indexed by timestamp.

    Raises:
        ValueError: If seqn is None or if no valid NHANES data is found.
    """

    if seqn is None:
        raise ValueError("The seqn is required for nhanes data")

    # list all files in directory starting with PAX
    pax_files = [f for f in os.listdir(file_dir) if f.startswith('PAX')]
    # for each file starting with PAXDAY check if PAXHD and PAXMIN are present
    versions = []
    for file in pax_files:
        if file.startswith('PAXDAY'):
            version = file.split("_")[1].strip('.xpt')
            if f'PAXHD_{version}.xpt' in pax_files and f'PAXMIN_{version}.xpt' in pax_files:
                if seqn in pd.read_sas(f"{file_dir}/PAXDAY_{version}.xpt")['SEQN'].unique():
                    versions.append(version)

    if verbose:
        print(f"Found {len(versions)} versions of NHANES data")

    if len(versions) == 0:
        raise ValueError(f"No valid versions of NHANES data found - this might be due to missing files. For each version we expect to find PAXDAY, PAXHD and PAXMIN files.")

    # read all day-level files
    day_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading day-level files"):
        curr = pd.read_sas(f"{file_dir}/PAXDAY_{version}.xpt")
        curr = curr[curr['SEQN'] == seqn]
        day_x = pd.concat([day_x, curr], ignore_index=True)

    if day_x.empty:
        raise ValueError(f"No day-level data found for person {seqn}")

    # rename columns
    day_x = day_x.rename(columns=str.lower)
    day_x = remove_bytes(day_x)

    if verbose:
        print(f"Read {day_x.shape[0]} day-level records for person {seqn}")

    # check data quality flags
    day_x = day_x[day_x['paxqfd'] < 1]

    # check if valid hours are greater than 16
    day_x = day_x.assign(valid_hours=(day_x['paxwwmd'] + day_x['paxswmd']) / 60)
    day_x = day_x[day_x['valid_hours'] > 16]

    # check if there are at least 4 days of data
    day_x = day_x.groupby('seqn').filter(lambda x: len(x) >= 4)

    # read all minute-level files
    min_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading minute-level files"):
        itr_x = pd.read_sas(f"{file_dir}/PAXMIN_{version}.xpt", chunksize=100000)
        for chunk in tqdm(itr_x, desc=f"Processing chunks for version {version}"):
            curr = clean_data(chunk, day_x)
            curr = curr[curr['SEQN'] == seqn]
            min_x = pd.concat([min_x, curr], ignore_index=True)

    min_x = min_x.rename(columns=str.lower)
    min_x = remove_bytes(min_x)

    if verbose:
        print(f"Read {min_x.shape[0]} minute-level records for person {seqn}")

    # add header data
    head_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading header files"):
        curr = pd.read_sas(f"{file_dir}/PAXHD_{version}.xpt")
        curr = curr[curr['SEQN'] == seqn]
        head_x = pd.concat([head_x, curr], ignore_index=True)

    head_x = head_x.rename(columns=str.lower)
    head_x = head_x[['seqn', 'paxftime', 'paxfday']].rename(columns={
        'paxftime': 'day1_start_time', 'paxfday': 'day1_which_day'
    })

    min_x = min_x.merge(head_x, on='seqn')
    min_x = remove_bytes(min_x)

    if verbose:
        print(f"Merged header and minute-level data for person {seqn}")

    # calculate measure time
    min_x['measure_time'] = min_x.apply(calculate_measure_time, axis=1)
    min_x['measure_hour'] = min_x['measure_time'].dt.hour

    valid_startend = min_x.groupby(['seqn', 'paxdaym']).agg(
        start=('measure_hour', 'min'),
        end=('measure_hour', 'max')
    ).reset_index()

    min_x = min_x.merge(valid_startend, on=['seqn', 'paxdaym'])
    min_x = min_x[(min_x['start'] == 0) & (min_x['end'] == 23)]

    min_x['measure_min'] = min_x['measure_time'].dt.minute
    min_x['myepoch'] = (12 * min_x['measure_hour'] + np.floor(min_x['measure_min'] / 5 + 1)).astype(int)

    # Count epochs per day and filter for complete days (288 epochs)
    epoch_counts = min_x.groupby(['seqn', 'paxdaym'])['myepoch'].nunique().reset_index()
    epoch_counts = epoch_counts[epoch_counts['myepoch'] == 288]
    min_x = min_x.merge(epoch_counts[['seqn', 'paxdaym']], on=['seqn', 'paxdaym'])

    # Count valid days per participant and filter for at least 4 valid days
    valid_days = min_x.groupby('seqn')['paxdaym'].unique().reset_index()
    valid_days = valid_days[valid_days['paxdaym'].apply(len) >= 4]
    min_x = min_x[min_x['seqn'].isin(valid_days['seqn'])]

    min_x = min_x.rename(columns={
        'paxmxm': 'X', 'paxmym': 'Y', 'paxmzm': 'Z', 'measure_time': 'TIMESTAMP', 
    })

    if verbose:
        print(f"Renamed columns and set timestamp index for person {seqn}")

    # set wear and sleep columns
    min_x['wear'] = min_x['paxpredm'].astype(int).isin([1, 2]).astype(int)
    min_x['sleep'] = min_x['paxpredm'].astype(int).isin([2]).astype(int)

    min_x.set_index('TIMESTAMP', inplace=True)
    min_x = min_x[['X', 'Y', 'Z', 'wear', 'sleep', 'paxpredm']]


    meta_dict['raw_n_datapoints'] = min_x.shape[0]
    meta_dict['raw_start_datetime'] = min_x.index.min()
    meta_dict['raw_end_datetime'] = min_x.index.max()
    meta_dict['raw_data_frequency'] = 'minute-level'
    meta_dict['raw_data_type'] = 'accelerometer'
    meta_dict['raw_data_unit'] = 'MIMS'

    if verbose:
        print(f"Loaded {min_x.shape[0]} minute-level Accelerometer records from {file_dir}")

    return min_x

def filter_and_preprocess_nhanes_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Filter NHANES accelerometer data for incomplete days and non-consecutive sequences.

    Args:
        data (pd.DataFrame): Raw NHANES accelerometer data
        meta_dict (dict, optional): Dictionary to store metadata. Defaults to {}.
        verbose (bool, optional): Whether to print processing status. Defaults to False.

    Returns:
        pd.DataFrame: Filtered accelerometer data containing only complete, consecutive days
    """
    _data = data.copy()
    
    old_n = _data.shape[0]
    _data = filter_incomplete_days(_data, data_freq=1/60)
    if verbose:
        print(f"Filtered out {old_n - data.shape[0]} minute-level ENMO records due to incomplete daily coverage")

    _data.index = pd.to_datetime(_data.index)
        
    old_n = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(f"Filtered out {old_n - _data.shape[0]} minute-level ENMO records due to filtering for longest consecutive sequence of days")

    meta_dict['n_days'] = len(np.unique(_data.index.date))

    _data[['X_raw', 'Y_raw', 'Z_raw']] = _data[['X', 'Y', 'Z']]
    _data[['X', 'Y', 'Z']] = _data[['X', 'Y', 'Z']] / 9.81 # convert from MIMS to aprrox. mg 
    _data['ENMO'] = calculate_enmo(_data) * 257 # factor of 257 as a result of parameter tuning for making cosinorage predictions match

    if verbose:
        print(f"Calculated ENMO data")

    return _data

def resample_nhanes_data(data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False) -> pd.DataFrame:
    """
    Resample NHANES accelerometer data to 1-minute intervals using linear interpolation.

    Args:
        data (pd.DataFrame): NHANES accelerometer data
        meta_dict (dict, optional): Dictionary to store metadata. Defaults to {}.
        verbose (bool, optional): Whether to print processing status. Defaults to False.

    Returns:
        pd.DataFrame: Resampled accelerometer data with 1-minute intervals
    """
    _data = data.copy()

    _data = _data.resample('1min').interpolate(method='linear').bfill()
    _data['sleep'] = _data['sleep'].round(0)
    _data['wear'] = _data['wear'].round(0)

    if verbose:
        print(f"Resampled {data.shape[0]} to {_data.shape[0]} timestamps")

    return _data

def remove_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert byte string columns to regular strings in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing potential byte string columns

    Returns:
        pd.DataFrame: DataFrame with byte strings converted to UTF-8 strings
    """
    for col in df.select_dtypes([object]):  # Select columns with object type (likely byte strings)
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df

def clean_data(df: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NHANES minute-level data by applying quality filters.

    Args:
        df (pd.DataFrame): Raw minute-level NHANES data
        days (pd.DataFrame): Day-level NHANES data for filtering

    Returns:
        pd.DataFrame: Cleaned minute-level data excluding invalid measurements and participants
    """
    df = df[df['SEQN'].isin(days['seqn'])]
    df = df[df['PAXMTSM'] != -0.01]
    df = df[~df['PAXPREDM'].isin([3, 4])]
    df = df[df['PAXQFM'] < 1]
    return df

def calculate_measure_time(row):
    """
    Calculate the measurement timestamp for a row of NHANES data.

    Args:
        row (pd.Series): Row containing 'day1_start_time' and 'paxssnmp' values

    Returns:
        datetime: Calculated measurement timestamp
    """
    base_time = datetime.strptime(row['day1_start_time'], "%H:%M:%S")
    measure_time = base_time + timedelta(seconds=row['paxssnmp'] / 80)
    return measure_time