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
from typing import Optional

from .frequency_detection import detect_frequency_from_timestamps
from .filtering import filter_incomplete_days, filter_consecutive_days


def read_generic_xD(file_path: str, meta_dict: dict, n_dimensions: int, time_column: str = 'timestamp', data_columns: Optional[list] = None, verbose: bool = False):
    """
    Read generic accelerometer or count data from a CSV file.
    
    This function loads data from a CSV file and standardizes the column names
    for further processing. It supports both 1-dimensional (counts/ENMO) and
    3-dimensional (accelerometer) data formats.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the data.
    meta_dict : dict
        Dictionary to store metadata about the loaded data. Will be populated with:
        - raw_n_datapoints: Number of data points
        - raw_start_datetime: Start timestamp
        - raw_end_datetime: End timestamp
        - sf: Sampling frequency in Hz
        - raw_data_frequency: Sampling frequency as string
        - raw_data_type: Type of data ('Counts' or 'Accelerometer')
        - raw_data_unit: Unit of data ('counts' or 'mg')
    n_dimensions : int
        Number of dimensions in the data. Must be either 1 (for counts/ENMO) or 3 (for accelerometer).
    time_column : str, default='timestamp'
        Name of the timestamp column in the CSV file.
    data_columns : list, optional
        Names of the data columns in the CSV file. If not provided, defaults are:
        - ['counts'] for n_dimensions=1
        - ['x', 'y', 'z'] for n_dimensions=3
    verbose : bool, default=False
        Whether to print progress information.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data with standardized column names:
        - For n_dimensions=1: ['ENMO'] (single column)
        - For n_dimensions=3: ['x', 'y', 'z'] (three columns)
        The DataFrame has a datetime index from the timestamp column.
    
    Raises
    ------
    ValueError
        If n_dimensions is not 1 or 3, or if the number of data_columns doesn't match n_dimensions.
    
    Examples
    --------
    Load 1-dimensional count data:
    
    >>> meta_dict = {}
    >>> data = read_generic_xD(
    ...     file_path='data/counts.csv',
    ...     meta_dict=meta_dict,
    ...     n_dimensions=1,
    ...     time_column='time',
    ...     data_columns=['counts']
    ... )
    >>> print(data.columns)
    Index(['ENMO'], dtype='object')
    
    Load 3-dimensional accelerometer data:
    
    >>> meta_dict = {}
    >>> data = read_generic_xD(
    ...     file_path='data/accel.csv',
    ...     meta_dict=meta_dict,
    ...     n_dimensions=3,
    ...     time_column='timestamp',
    ...     data_columns=['accel_x', 'accel_y', 'accel_z']
    ... )
    >>> print(data.columns)
    Index(['x', 'y', 'z'], dtype='object')
    
    Notes
    -----
    The function automatically:
    - Converts timestamps to datetime objects
    - Removes timezone information
    - Fills missing values with 0
    - Sorts data by timestamp
    - Detects sampling frequency from timestamps
    - Populates metadata dictionary with data information
    """
    
    if n_dimensions not in [1, 3]:
        raise ValueError("n_dimensions must be either 1 or 3")
    
    if data_columns is not None:
        if n_dimensions != len(data_columns):
            raise ValueError("n_dimensions must be equal to the number of data columns")

    data = pd.read_csv(file_path)

    if verbose:
        print(f"Read csv file from {file_path}")

    # Set default data_columns if not provided
    if data_columns is None:
        if n_dimensions == 1:
            data_columns = ['counts']
        elif n_dimensions == 3:
            data_columns = ['x', 'y', 'z']
        else:
            raise ValueError("n_dimensions must be either 1 or 3")

    # Rename columns to standard format
    column_mapping = {time_column: 'TIMESTAMP'}
    if n_dimensions == 1:
        column_mapping[data_columns[0]] = 'ENMO'
    elif n_dimensions == 3:
        column_mapping[data_columns[0]] = 'x'
        column_mapping[data_columns[1]] = 'y'
        column_mapping[data_columns[2]] = 'z'
    else:
        raise ValueError("n_dimensions must be either 1 or 3")
    
    data = data.rename(columns=column_mapping)
    
    # Convert UTC timestamps to local time
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP']).dt.tz_localize(None)
    data.set_index('TIMESTAMP', inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(f"Loaded {data.shape[0]} Count data records from {file_path}")

    meta_dict['raw_n_datapoints'] = data.shape[0]
    meta_dict['raw_start_datetime'] = data.index.min()
    meta_dict['raw_end_datetime'] = data.index.max()
    meta_dict['sf'] = detect_frequency_from_timestamps(data.index)
    meta_dict['raw_data_frequency'] = f'{meta_dict["sf"]}Hz'
    meta_dict['raw_data_type'] = 'Counts'
    meta_dict['raw_data_unit'] = 'counts'

    return data