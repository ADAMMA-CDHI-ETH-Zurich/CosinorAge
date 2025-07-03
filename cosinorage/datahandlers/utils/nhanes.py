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

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .calc_enmo import calculate_enmo
from .filtering import filter_consecutive_days, filter_incomplete_days


def read_nhanes_data(
    file_dir: str,
    seqn: Optional[str] = None,
    meta_dict: dict = {},
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read and process NHANES accelerometer data files for a specific person.

    This function loads and processes National Health and Nutrition Examination Survey (NHANES)
    accelerometer data for a specific participant. It handles the complex NHANES data structure
    including day-level, minute-level, and header files.

    Parameters
    ----------
    file_dir : str
        Directory containing NHANES data files. Must contain:
        - PAXDAY_*.xpt: Day-level data files
        - PAXHD_*.xpt: Header data files
        - PAXMIN_*.xpt: Minute-level data files
    seqn : str, optional
        Unique identifier for the participant. Required for data extraction.
    meta_dict : dict, default={}
        Dictionary to store metadata about the loaded data. Will be populated with:
        - raw_n_datapoints: Number of data points
        - raw_start_datetime: Start timestamp
        - raw_end_datetime: End timestamp
        - raw_data_frequency: Sampling frequency ('minute-level')
        - raw_data_type: Type of data ('accelerometer')
        - raw_data_unit: Unit of data ('MIMS')
    verbose : bool, default=False
        Whether to print processing status and progress information.

    Returns
    -------
    pd.DataFrame
        Processed accelerometer data with columns:
        - 'x', 'y', 'z': Accelerometer values in MIMS units
        - 'wear': Binary wear detection (1=worn, 0=not worn)
        - 'sleep': Binary sleep detection (1=sleep, 0=wake)
        - 'paxpredm': Original NHANES prediction values
        The DataFrame is indexed by timestamp.

    Raises
    ------
    ValueError
        If seqn is None or if no valid NHANES data is found for the participant.

    Notes
    -----
    - Automatically detects and processes multiple NHANES data versions
    - Applies data quality filters (paxqfd < 1, valid_hours > 16)
    - Requires at least 4 days of valid data per participant
    - Filters for complete days (288 epochs per day)
    - Converts column names to lowercase for consistency
    - Removes byte-encoded data using remove_bytes function

    Examples
    --------
    >>> import os
    >>>
    >>> # Load NHANES data for a specific participant
    >>> file_dir = '/path/to/nhanes/data'
    >>> seqn = '12345'  # Participant ID
    >>> meta_dict = {}
    >>> data = read_nhanes_data(
    ...     file_dir=file_dir,
    ...     seqn=seqn,
    ...     meta_dict=meta_dict,
    ...     verbose=True
    ... )
    >>> print(f"Loaded {len(data)} records for participant {seqn}")
    >>> print(f"Data columns: {data.columns.tolist()}")
    """

    if seqn is None:
        raise ValueError("The seqn is required for nhanes data")

    # list all files in directory starting with PAX
    pax_files = [f for f in os.listdir(file_dir) if f.startswith("PAX")]
    # for each file starting with PAXDAY check if PAXHD and PAXMIN are present
    versions = []
    for file in pax_files:
        if file.startswith("PAXDAY"):
            version = file.split("_")[1].strip(".xpt")
            if (
                f"PAXHD_{version}.xpt" in pax_files
                and f"PAXMIN_{version}.xpt" in pax_files
            ):
                if (
                    seqn
                    in pd.read_sas(f"{file_dir}/PAXDAY_{version}.xpt")[
                        "SEQN"
                    ].unique()
                ):
                    versions.append(version)

    if verbose:
        print(f"Found {len(versions)} versions of NHANES data: {versions}")

    if len(versions) == 0:
        raise ValueError(
            f"No valid versions of NHANES data found - this might be due to missing files. For each version we expect to find PAXDAY, PAXHD and PAXMIN files."
        )

    # read all day-level files
    day_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading day-level files"):
        curr = pd.read_sas(f"{file_dir}/PAXDAY_{version}.xpt")
        curr = curr[curr["SEQN"] == seqn]
        day_x = pd.concat([day_x, curr], ignore_index=True)

    if day_x.empty:
        raise ValueError(f"No day-level data found for person {seqn}")

    # rename columns
    day_x = day_x.rename(columns=str.lower)
    day_x = remove_bytes(day_x)

    if verbose:
        print(f"Read {day_x.shape[0]} day-level records for person {seqn}")

    # check data quality flags
    day_x = day_x[day_x["paxqfd"] < 1]

    # check if valid hours are greater than 16
    day_x = day_x.assign(
        valid_hours=(day_x["paxwwmd"] + day_x["paxswmd"]) / 60
    )
    day_x = day_x[day_x["valid_hours"] > 16]

    # check if there are at least 4 days of data
    day_x = day_x.groupby("seqn").filter(lambda x: len(x) >= 4)

    # read all minute-level files
    min_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading minute-level files"):
        itr_x = pd.read_sas(
            f"{file_dir}/PAXMIN_{version}.xpt", chunksize=100000
        )
        for chunk in tqdm(
            itr_x, desc=f"Processing chunks for version {version}"
        ):
            curr = clean_data(chunk, day_x)
            curr = curr[curr["SEQN"] == seqn]
            min_x = pd.concat([min_x, curr], ignore_index=True)

    min_x = min_x.rename(columns=str.lower)
    min_x = remove_bytes(min_x)

    if verbose:
        print(f"Read {min_x.shape[0]} minute-level records for person {seqn}")

    # add header data
    head_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading header files"):
        curr = pd.read_sas(f"{file_dir}/PAXHD_{version}.xpt")
        curr = curr[curr["SEQN"] == seqn]
        head_x = pd.concat([head_x, curr], ignore_index=True)

    head_x = head_x.rename(columns=str.lower)
    head_x = head_x[["seqn", "paxftime", "paxfday"]].rename(
        columns={"paxftime": "day1_start_time", "paxfday": "day1_which_day"}
    )

    min_x = min_x.merge(head_x, on="seqn")
    min_x = remove_bytes(min_x)

    if verbose:
        print(f"Merged header and minute-level data for person {seqn}")

    # calculate measure time
    min_x["measure_time"] = min_x.apply(calculate_measure_time, axis=1)
    min_x["measure_hour"] = min_x["measure_time"].dt.hour

    valid_startend = (
        min_x.groupby(["seqn", "paxdaym"])
        .agg(start=("measure_hour", "min"), end=("measure_hour", "max"))
        .reset_index()
    )

    min_x = min_x.merge(valid_startend, on=["seqn", "paxdaym"])
    min_x = min_x[(min_x["start"] == 0) & (min_x["end"] == 23)]

    min_x["measure_min"] = min_x["measure_time"].dt.minute
    min_x["myepoch"] = (
        12 * min_x["measure_hour"] + np.floor(min_x["measure_min"] / 5 + 1)
    ).astype(int)

    # Count epochs per day and filter for complete days (288 epochs)
    epoch_counts = (
        min_x.groupby(["seqn", "paxdaym"])["myepoch"].nunique().reset_index()
    )
    epoch_counts = epoch_counts[epoch_counts["myepoch"] == 288]
    min_x = min_x.merge(
        epoch_counts[["seqn", "paxdaym"]], on=["seqn", "paxdaym"]
    )

    # Count valid days per participant and filter for at least 4 valid days
    valid_days = min_x.groupby("seqn")["paxdaym"].unique().reset_index()
    valid_days = valid_days[valid_days["paxdaym"].apply(len) >= 4]
    min_x = min_x[min_x["seqn"].isin(valid_days["seqn"])]

    min_x = min_x.rename(
        columns={
            "paxmxm": "x",
            "paxmym": "y",
            "paxmzm": "z",
            "measure_time": "timestamp",
        }
    )

    if verbose:
        print(f"Renamed columns and set timestamp index for person {seqn}")

    # set wear and sleep columns
    min_x["wear"] = min_x["paxpredm"].astype(int).isin([1, 2]).astype(int)
    min_x["sleep"] = min_x["paxpredm"].astype(int).isin([2]).astype(int)

    min_x.set_index("timestamp", inplace=True)
    min_x = min_x[["x", "y", "z", "wear", "sleep", "paxpredm"]]

    meta_dict["raw_n_datapoints"] = min_x.shape[0]
    meta_dict["raw_start_datetime"] = min_x.index.min()
    meta_dict["raw_end_datetime"] = min_x.index.max()
    meta_dict["raw_data_frequency"] = "minute-level"
    meta_dict["raw_data_type"] = "accelerometer"
    meta_dict["raw_data_unit"] = "MIMS"

    if verbose:
        print(
            f"Loaded {min_x.shape[0]} minute-level Accelerometer records from {file_dir}"
        )

    return min_x


def filter_and_preprocess_nhanes_data(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Filter NHANES accelerometer data for incomplete days and non-consecutive sequences.

    This function applies data quality filters to NHANES accelerometer data and
    converts the data to the standard format used by the CosinorAge pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Raw NHANES accelerometer data with columns ['x', 'y', 'z', 'wear', 'sleep', 'paxpredm']
        and datetime index.
    meta_dict : dict, default={}
        Dictionary to store metadata about the filtering process. Will be populated with:
        - n_days: Number of valid days after filtering
    verbose : bool, default=False
        Whether to print processing status and progress information.

    Returns
    -------
    pd.DataFrame
        Filtered and preprocessed accelerometer data with columns:
        - 'x', 'y', 'z': Accelerometer values converted from MIMS to mg units
        - 'x_raw', 'y_raw', 'z_raw': Original accelerometer values
        - 'wear': Binary wear detection
        - 'sleep': Binary sleep detection
        - 'paxpredm': Original NHANES prediction values
        - 'ENMO': Calculated ENMO values (scaled by factor of 257)

    Notes
    -----
    - Removes incomplete days using filter_incomplete_days
    - Selects longest consecutive sequence using filter_consecutive_days
    - Converts accelerometer values from MIMS to mg units (division by 9.81)
    - Calculates ENMO values with a scaling factor of 257 for parameter tuning
    - Stores original values in *_raw columns for reference

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample NHANES data
    >>> dates = pd.date_range('2023-01-01', periods=10000, freq='min')
    >>> data = pd.DataFrame({
    ...     'x': np.random.randn(10000),
    ...     'y': np.random.randn(10000),
    ...     'z': np.random.randn(10000),
    ...     'wear': np.random.choice([0, 1], 10000),
    ...     'sleep': np.random.choice([0, 1], 10000),
    ...     'paxpredm': np.random.choice([0, 1, 2], 10000)
    ... }, index=dates)
    >>>
    >>> # Filter and preprocess the data
    >>> meta_dict = {}
    >>> processed_data = filter_and_preprocess_nhanes_data(
    ...     data, meta_dict=meta_dict, verbose=True
    ... )
    >>> print(f"Processed data shape: {processed_data.shape}")
    >>> print(f"Number of days: {meta_dict.get('n_days', 'N/A')}")
    """
    _data = data.copy()

    old_n = _data.shape[0]
    _data = filter_incomplete_days(_data, data_freq=1 / 60)
    if verbose:
        print(
            f"Filtered out {old_n - data.shape[0]} minute-level ENMO records due to incomplete daily coverage"
        )

    _data.index = pd.to_datetime(_data.index)

    old_n = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(
            f"Filtered out {old_n - _data.shape[0]} minute-level ENMO records due to filtering for longest consecutive sequence of days"
        )

    meta_dict["n_days"] = len(np.unique(_data.index.date))

    _data[["x_raw", "y_raw", "z_raw"]] = _data[["x", "y", "z"]]
    _data[["x", "y", "z"]] = (
        _data[["x", "y", "z"]] / 9.81
    )  # convert from MIMS to aprrox. mg
    _data["enmo"] = (
        calculate_enmo(_data) * 257
    )  # factor of 257 as a result of parameter tuning for making cosinorage predictions match

    if verbose:
        print(f"Calculated ENMO data")

    return _data


def resample_nhanes_data(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Resample NHANES accelerometer data to 1-minute intervals using linear interpolation.

    This function ensures consistent minute-level resolution for NHANES accelerometer data
    by resampling to 1-minute intervals and handling categorical variables appropriately.

    Parameters
    ----------
    data : pd.DataFrame
        NHANES accelerometer data with datetime index and columns including
        'x', 'y', 'z', 'sleep', 'wear'.
    meta_dict : dict, default={}
        Dictionary to store metadata about the resampling process.
    verbose : bool, default=False
        Whether to print processing status and progress information.

    Returns
    -------
    pd.DataFrame
        Resampled accelerometer data with consistent 1-minute intervals.
        Categorical variables ('sleep', 'wear') are rounded to nearest integer.

    Notes
    -----
    - Uses pandas resample('1min') with linear interpolation for continuous variables
    - Applies forward fill (bfill) to handle any remaining gaps
    - Rounds categorical variables ('sleep', 'wear') to nearest integer
    - Maintains data integrity for binary classification variables

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample NHANES data with irregular intervals
    >>> dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:30',
    ...                         '2023-01-01 00:03:00', '2023-01-01 00:04:30'])
    >>> data = pd.DataFrame({
    ...     'x': [0.1, 0.2, 0.3, 0.4],
    ...     'y': [0.1, 0.2, 0.3, 0.4],
    ...     'z': [0.1, 0.2, 0.3, 0.4],
    ...     'sleep': [0, 1, 0, 1],
    ...     'wear': [1, 1, 0, 1]
    ... }, index=dates)
    >>>
    >>> # Resample to minute level
    >>> meta_dict = {}
    >>> resampled_data = resample_nhanes_data(data, meta_dict=meta_dict, verbose=True)
    >>> print(f"Original data points: {len(data)}")
    >>> print(f"Resampled data points: {len(resampled_data)}")
    """
    _data = data.copy()

    _data = _data.resample("1min").mean().interpolate(method="linear").bfill()
    _data["sleep"] = _data["sleep"].round(0)
    _data["wear"] = _data["wear"].round(0)

    if verbose:
        print(f"Resampled {data.shape[0]} to {_data.shape[0]} timestamps")

    return _data


def remove_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert byte string columns to regular strings in a DataFrame.

    This function handles byte-encoded string columns that are common in NHANES data
    files, converting them to UTF-8 encoded strings for proper processing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing potential byte string columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with byte strings converted to UTF-8 strings.
        Non-byte string columns remain unchanged.

    Notes
    -----
    - Only processes columns with object dtype (likely to contain byte strings)
    - Uses UTF-8 encoding for conversion
    - Leaves non-byte string values unchanged
    - Common in NHANES data due to SAS file format

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample DataFrame with byte strings
    >>> data = {
    ...     'col1': [b'hello', b'world', 'normal_string'],
    ...     'col2': [1, 2, 3],
    ...     'col3': [b'byte1', b'byte2', b'byte3']
    ... }
    >>> df = pd.DataFrame(data)
    >>>
    >>> # Convert byte strings
    >>> cleaned_df = remove_bytes(df)
    >>> print(cleaned_df['col1'].iloc[0])  # 'hello' instead of b'hello'
    """
    for col in df.select_dtypes(
        [object]
    ):  # Select columns with object type (likely byte strings)
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )
    return df


def clean_data(df: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NHANES minute-level data by applying quality filters.

    This function applies multiple quality filters to NHANES minute-level data
    to ensure only valid measurements are included in the analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Raw minute-level NHANES data containing columns:
        - 'SEQN': Participant identifier
        - 'PAXMTSM': Minute-level timestamp
        - 'PAXPREDM': Prediction values
        - 'PAXQFM': Quality flag
    days : pd.DataFrame
        Day-level NHANES data containing valid participant identifiers in 'seqn' column.

    Returns
    -------
    pd.DataFrame
        Cleaned minute-level data with invalid measurements and participants removed.

    Notes
    -----
    - Filters for participants present in day-level data
    - Removes measurements with PAXMTSM = -0.01 (invalid timestamp)
    - Excludes PAXPREDM values of 3 or 4 (invalid predictions)
    - Removes measurements with PAXQFM >= 1 (poor quality)

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample NHANES data
    >>> minute_data = pd.DataFrame({
    ...     'SEQN': ['12345', '12345', '12346', '12345'],
    ...     'PAXMTSM': [0, -0.01, 60, 120],
    ...     'PAXPREDM': [1, 2, 3, 1],
    ...     'PAXQFM': [0, 0, 1, 0]
    ... })
    >>>
    >>> day_data = pd.DataFrame({'seqn': ['12345']})
    >>>
    >>> # Clean the data
    >>> cleaned_data = clean_data(minute_data, day_data)
    >>> print(f"Original records: {len(minute_data)}")
    >>> print(f"Cleaned records: {len(cleaned_data)}")
    """
    df = df[df["SEQN"].isin(days["seqn"])]
    df = df[df["PAXMTSM"] != -0.01]
    df = df[~df["PAXPREDM"].isin([3, 4])]
    df = df[df["PAXQFM"] < 1]
    return df


def calculate_measure_time(row):
    """
    Calculate the measurement timestamp for a row of NHANES data.

    This function converts NHANES timing information into actual datetime timestamps
    by combining the day start time with the seconds since midnight.

    Parameters
    ----------
    row : pd.Series
        Row containing timing information:
        - 'day1_start_time': Start time of the first day in format "HH:MM:SS"
        - 'paxssnmp': Seconds since midnight (scaled by 80)

    Returns
    -------
    datetime
        Calculated measurement timestamp combining base time and offset.

    Notes
    -----
    - Converts day1_start_time string to datetime object
    - Divides paxssnmp by 80 to get actual seconds (NHANES scaling factor)
    - Adds the offset to the base time to get measurement timestamp
    - Used for creating proper datetime index for NHANES data

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample row with timing information
    >>> row = pd.Series({
    ...     'day1_start_time': '08:30:00',
    ...     'paxssnmp': 8000  # 100 seconds * 80
    ... })
    >>>
    >>> # Calculate measurement time
    >>> measure_time = calculate_measure_time(row)
    >>> print(f"Measurement time: {measure_time}")
    >>> # Output: 1900-01-01 08:31:40 (base time + 100 seconds)
    """
    base_time = datetime.strptime(row["day1_start_time"], "%H:%M:%S")
    measure_time = base_time + timedelta(seconds=row["paxssnmp"] / 80)
    return measure_time
