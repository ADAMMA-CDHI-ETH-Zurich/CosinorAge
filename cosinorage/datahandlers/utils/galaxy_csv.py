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

from typing import Optional

import pandas as pd

from .filtering import filter_consecutive_days, filter_incomplete_days
from .frequency_detection import detect_frequency_from_timestamps


def read_galaxy_csv_data(
    galaxy_file_path: str,
    meta_dict: dict,
    time_column: str = "timestamp",
    data_columns: Optional[list] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read ENMO data from Galaxy Watch CSV file.

    This function loads ENMO (Euclidean Norm Minus One) data from Samsung Galaxy Watch
    CSV files and standardizes the format for further processing.

    Parameters
    ----------
    galaxy_file_path : str
        Path to the Galaxy Watch CSV data file containing ENMO values.
    meta_dict : dict
        Dictionary to store metadata about the loaded data. Will be populated with:
        - raw_n_datapoints: Number of data points
        - raw_start_datetime: Start timestamp
        - raw_end_datetime: End timestamp
        - sf: Sampling frequency in Hz
        - raw_data_frequency: Sampling frequency as string
        - raw_data_type: Type of data ('ENMO')
        - raw_data_unit: Unit of data ('mg')
    time_column : str, default='timestamp'
        Name of the timestamp column in the CSV file.
    data_columns : list, optional
        Names of the data columns in the CSV file. If not provided, defaults to ['enmo'].
    verbose : bool, default=False
        Whether to print progress information during loading.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ENMO data with standardized column names:
        - 'ENMO': ENMO values in mg units
        The DataFrame has a datetime index from the timestamp column.

    Notes
    -----
    - The function automatically converts UTC timestamps to local time
    - Missing values are filled with 0
    - Data is sorted by timestamp
    - Sampling frequency is automatically detected from timestamps
    - Column names are standardized to 'ENMO' for consistency

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Load ENMO data from Galaxy Watch CSV file
    >>> meta_dict = {}
    >>> data = read_galaxy_csv_data(
    ...     galaxy_file_path='data/galaxy_enmo.csv',
    ...     meta_dict=meta_dict,
    ...     time_column='time',
    ...     data_columns=['enmo_mg'],
    ...     verbose=True
    ... )
    >>> print(f"Loaded {len(data)} ENMO records")
    >>> print(f"Sampling frequency: {meta_dict['sf']:.1f} Hz")
    """

    data = pd.read_csv(galaxy_file_path)

    if verbose:
        print(f"Read csv file from {galaxy_file_path}")

    # Set default data_columns if not provided
    if data_columns is None:
        data_columns = ["enmo"]

    # Rename columns to standard format
    column_mapping = {time_column: "timestamp"}
    for i, col in enumerate(data_columns):
        if i == 0:  # First column should be ENMO
            column_mapping[col] = "enmo"

    data = data.rename(columns=column_mapping)

    # Convert UTC timestamps to local time
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_localize(None)
    data.set_index("timestamp", inplace=True)

    data = data.fillna(0)
    data.sort_index(inplace=True)

    if verbose:
        print(
            f"Loaded {data.shape[0]} ENMO data records from {galaxy_file_path}"
        )

    meta_dict["raw_n_datapoints"] = data.shape[0]
    meta_dict["raw_start_datetime"] = data.index.min()
    meta_dict["raw_end_datetime"] = data.index.max()
    meta_dict["sf"] = detect_frequency_from_timestamps(pd.Series(data.index))
    meta_dict["raw_data_frequency"] = f'{meta_dict["sf"]:.3g}Hz'
    meta_dict["raw_data_type"] = "ENMO"
    meta_dict["raw_data_unit"] = "mg"

    return data


def filter_galaxy_csv_data(
    data: pd.DataFrame,
    meta_dict: dict = {},
    verbose: bool = False,
    preprocess_args: dict = {},
) -> pd.DataFrame:
    """
    Filter Galaxy Watch ENMO data by removing incomplete days and selecting longest consecutive sequence.

    This function applies data quality filters to Galaxy Watch ENMO data, including
    removal of incomplete days and selection of the longest consecutive sequence of days.

    Parameters
    ----------
    data : pd.DataFrame
        Raw ENMO data with datetime index and 'ENMO' column.
    meta_dict : dict, default={}
        Dictionary to store metadata about the filtering process. Should contain:
        - sf: Sampling frequency in Hz
    verbose : bool, default=False
        Whether to print progress information during filtering.
    preprocess_args : dict, default={}
        Dictionary containing filtering parameters:
        - required_daily_coverage: Minimum fraction of daily data required (default: 0.5)

    Returns
    -------
    pd.DataFrame
        Filtered ENMO data containing only complete and consecutive days.
        The DataFrame maintains the same structure as the input.

    Notes
    -----
    - Removes days that don't meet the required daily coverage threshold
    - Selects the longest sequence of consecutive days (minimum 4 days required)
    - Resamples data to minute-level resolution
    - Removes incomplete first and last days
    - Updates metadata with information about the filtering process

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample ENMO data
    >>> dates = pd.date_range('2023-01-01', periods=10000, freq='min')
    >>> data = pd.DataFrame({'ENMO': np.random.randn(10000)}, index=dates)
    >>>
    >>> # Filter the data
    >>> meta_dict = {'sf': 1/60}  # 1 sample per minute
    >>> preprocess_args = {'required_daily_coverage': 0.8}
    >>> filtered_data = filter_galaxy_csv_data(
    ...     data, meta_dict=meta_dict, preprocess_args=preprocess_args, verbose=True
    ... )
    >>> print(f"Original data points: {len(data)}")
    >>> print(f"Filtered data points: {len(filtered_data)}")
    """
    _data = data.copy()

    # filter out sparse days
    required_points_per_day = (
        preprocess_args.get("required_daily_coverage", 0.5) * 1440
    )
    n_old = _data.shape[0]
    _data = filter_incomplete_days(
        _data,
        data_freq=meta_dict["sf"],
        expected_points_per_day=required_points_per_day,
    )
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} ENMO records due to incomplete daily coverage"
        )

    # filter for longest consecutive sequence of days
    n_old = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} ENMO records due to filtering for longest consecutive sequence of days"
        )

    # resample to minute-level
    _data = _data.resample("1min").mean().interpolate(method="linear").bfill()
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
        print(
            f"Filtered out {n_old - _data.shape[0]}/{n_old} ENMO records due to filtering out first and last day"
        )

    return _data


def resample_galaxy_csv_data(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Ensure we have minute-level data across the whole timeseries.

    This function resamples Galaxy Watch ENMO data to ensure consistent
    minute-level resolution across the entire time series.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered ENMO data with datetime index and 'ENMO' column.
    meta_dict : dict, default={}
        Dictionary to store metadata about the resampling process.
    verbose : bool, default=False
        Whether to print progress information during resampling.

    Returns
    -------
    pd.DataFrame
        Resampled ENMO data with consistent minute-level resolution.
        The DataFrame maintains the same structure as the input.

    Notes
    -----
    - Uses pandas resample('1min') with linear interpolation
    - Forward fills any remaining gaps with bfill()
    - Ensures consistent temporal resolution for analysis
    - Updates metadata with information about the resampling process

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample ENMO data with irregular intervals
    >>> dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:30',
    ...                         '2023-01-01 00:03:00', '2023-01-01 00:04:30'])
    >>> data = pd.DataFrame({'ENMO': [0.1, 0.2, 0.3, 0.4]}, index=dates)
    >>>
    >>> # Resample to minute level
    >>> meta_dict = {}
    >>> resampled_data = resample_galaxy_csv_data(data, meta_dict=meta_dict, verbose=True)
    >>> print(f"Original data points: {len(data)}")
    >>> print(f"Resampled data points: {len(resampled_data)}")
    """
    _data = data.copy()

    n_old = _data.shape[0]
    _data = _data.resample("1min").mean().interpolate(method="linear").bfill()
    if verbose:
        print(f"Resampled {n_old} to {_data.shape[0]} timestamps")

    return _data


def preprocess_galaxy_csv_data(
    data: pd.DataFrame,
    preprocess_args: dict = {},
    meta_dict: dict = {},
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess Galaxy Watch ENMO data including rescaling, calibration, noise removal, and wear detection.

    This function applies preprocessing steps to Galaxy Watch ENMO data. Currently,
    wear detection is not implemented for ENMO data as the algorithm relies on
    raw accelerometer data.

    Parameters
    ----------
    data : pd.DataFrame
        Resampled ENMO data with datetime index and 'ENMO' column.
    preprocess_args : dict, default={}
        Dictionary containing preprocessing parameters (currently not used for ENMO data).
    meta_dict : dict, default={}
        Dictionary to store metadata about the preprocessing process.
    verbose : bool, default=False
        Whether to print progress information during preprocessing.

    Returns
    -------
    pd.DataFrame
        Preprocessed ENMO data with additional columns:
        - 'ENMO': Original ENMO values
        - 'wear': Wear detection column (set to -1 for ENMO data)

    Notes
    -----
    - Wear detection is not implemented for ENMO data
    - The 'wear' column is set to -1 to indicate no wear detection
    - Future implementations may add wear detection for ENMO data
    - The function maintains the original ENMO values

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample ENMO data
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    >>> data = pd.DataFrame({'ENMO': np.random.uniform(0, 0.1, 1440)}, index=dates)
    >>>
    >>> # Preprocess the data
    >>> meta_dict = {}
    >>> preprocess_args = {}
    >>> processed_data = preprocess_galaxy_csv_data(
    ...     data, preprocess_args=preprocess_args, meta_dict=meta_dict, verbose=True
    ... )
    >>> print(f"Processed data shape: {processed_data.shape}")
    >>> print(f"Wear column present: {'wear' in processed_data.columns}")
    """
    _data = data.copy()

    # wear detection - not implemented for enmo data yet (current algorithm relies on accelerometer data)
    _data["wear"] = -1

    if verbose:
        print(f"Preprocessed ENMO data")

    return _data
