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

import glob
import os
from typing import Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .filtering import filter_consecutive_days, filter_incomplete_days


def read_ukb_data(
    qc_file_path: str,
    enmo_file_dir: str,
    eid: int,
    meta_dict: dict = {},
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[Any, Union[float, Any]]]:
    """
    Read and process UK Biobank accelerometer data for a specific participant.

    This function loads and processes UK Biobank accelerometer data for a specific participant,
    applying quality control checks and converting the data to a standardized format.

    Parameters
    ----------
    qc_file_path : str
        Path to the quality control CSV file containing participant metadata.
        Must contain columns: eid, acc_data_problem, acc_weartime, acc_calibration,
        acc_owndata, acc_interrupt_period.
    enmo_file_dir : str
        Directory containing the ENMO data files (OUT_*.csv format).
    eid : int
        Participant ID to process.
    meta_dict : dict, default={}
        Dictionary to store metadata about the loaded data. Will be populated with:
        - raw_n_datapoints: Number of data points
        - raw_start_datetime: Start timestamp
        - raw_end_datetime: End timestamp
        - raw_data_frequency: Sampling frequency ('minute-level')
        - raw_data_type: Type of data ('ENMO')
        - raw_data_unit: Unit of data ('mg')
    verbose : bool, default=False
        Whether to print processing information and progress.

    Returns
    -------
    pd.DataFrame
        DataFrame containing processed ENMO data with columns:
        - 'ENMO': Euclidean Norm Minus One values in milligravity units
        The DataFrame has a datetime index.

    Raises
    ------
    FileNotFoundError
        If QC file or ENMO directory doesn't exist.
    ValueError
        If participant data is invalid or fails quality control checks.

    Notes
    -----
    - Applies multiple quality control filters from the QC file
    - Processes ENMO data from CSV files with acceleration headers
    - Converts timestamps to proper datetime format
    - Filters ENMO values >= 0.1, sets others to 0
    - Sorts data by timestamp for consistency

    Examples
    --------
    >>> import os
    >>>
    >>> # Load UK Biobank data for a specific participant
    >>> qc_file_path = '/path/to/ukb_qc.csv'
    >>> enmo_file_dir = '/path/to/enmo/files'
    >>> eid = 12345  # Participant ID
    >>> meta_dict = {}
    >>> data = read_ukb_data(
    ...     qc_file_path=qc_file_path,
    ...     enmo_file_dir=enmo_file_dir,
    ...     eid=eid,
    ...     meta_dict=meta_dict,
    ...     verbose=True
    ... )
    >>> print(f"Loaded {len(data)} ENMO records for participant {eid}")
    >>> print(f"Data range: {data.index.min()} to {data.index.max()}")
    """
    # check if qa_file_path and acc_file_path exist
    if not os.path.exists(qc_file_path):
        raise FileNotFoundError(f"QA file does not exist: {qc_file_path}")
    if not os.path.exists(enmo_file_dir):
        raise FileNotFoundError(
            f"ENMO file directory does not exist: {enmo_file_dir}"
        )

    qa_data = pd.read_csv(qc_file_path)

    if eid not in qa_data["eid"].values:
        raise ValueError(
            f"Eid {eid} not found in QA file - please try again with a different eid, e.g., {np.unique(qa_data['eid'].values)[:5]}"
        )

    acc_qc = qa_data[qa_data["eid"] == eid]

    # Exclude participants with data problems - filter rows where `acc_data_problem` is blank
    acc_qc = qa_data[
        qa_data["acc_data_problem"].isnull()
        | (qa_data["acc_data_problem"] == "")
    ]

    if acc_qc.empty:
        raise ValueError(
            f"Eid {eid} has no valid enmo data - check for data problems"
        )

    # Exclude participants with poor wear time - filter rows where `acc_weartime` is "Yes"
    acc_qc = acc_qc[acc_qc["acc_weartime"] == "Yes"]

    if acc_qc.empty:
        raise ValueError(
            f"Eid {eid} has no valid enmo data - check for wear time"
        )
    # Exclude participants with poor calibration - filter rows where `acc_calibration` is "Yes"
    acc_qc = acc_qc[acc_qc["acc_calibration"] == "Yes"]

    if acc_qc.empty:
        raise ValueError(
            f"Eid {eid} has no valid enmo data - check for calibration"
        )

    # Exclude participants not calibrated on their own data - filter rows where `acc_owndata` is "Yes"
    acc_qc = acc_qc[acc_qc["acc_owndata"] == "Yes"]

    if acc_qc.empty:
        raise ValueError(
            f"Eid {eid} has no valid enmo data - check for own data calibration"
        )

    # Exclude participants with interrupted recording periods - filter rows where `acc_interrupt_period` is 0
    acc_qc = acc_qc[acc_qc["acc_interrupt_period"] == 0]

    if acc_qc.empty:
        raise ValueError(
            f"Eid {eid} has no valid enmo data - check for interrupted recording periods"
        )

    if verbose:
        print(f"Quality control passed for eid {eid}")

    # read acc file
    enmo_file_names = glob.glob(os.path.join(enmo_file_dir, "OUT_*.csv"))

    data = pd.DataFrame()

    for file in tqdm(enmo_file_names):
        result = pd.read_csv(file, dtype={"eid": int}, low_memory=False)

        if eid in result["eid"].unique():
            result = result[result["eid"] == eid]

            # Filter rows containing "acceleration" and extract date, time, and other metadata
            save_date = result[
                result["enmo_mg"].str.contains("acceleration", na=False)
            ].copy()
            save_date["first_date"] = pd.to_datetime(
                save_date["enmo_mg"].str[20:30], errors="coerce"
            )
            save_date["start_time"] = pd.to_datetime(
                save_date["enmo_mg"].str[31:39],
                format="%H:%M:%S",
                errors="coerce",
            ).dt.time
            save_date["last_date"] = pd.to_datetime(
                save_date["enmo_mg"].str[42:52], errors="coerce"
            )
            save_date["end_time"] = pd.to_datetime(
                save_date["enmo_mg"].str[53:61],
                format="%H:%M:%S",
                errors="coerce",
            ).dt.time
            save_date["start_timestamp"] = (
                save_date["first_date"].astype(str)
                + " "
                + save_date["start_time"].astype(str)
            )
            save_date["eid"] = save_date["eid"].astype(str)

            # Keep only required columns
            save_date = save_date[
                [
                    "eid",
                    "start_timestamp",
                    "first_date",
                    "start_time",
                    "last_date",
                    "end_time",
                ]
            ]

            # Extract accelerometer data without headers
            ukb = result[
                ~result["enmo_mg"].str.contains("acceleration", na=False)
            ].copy()
            ukb["enmo_mg"] = pd.to_numeric(ukb["enmo_mg"], errors="coerce")

            # Add frequency column
            ukb["freq"] = ukb.groupby("eid").cumcount() + 1

            # Merge with `save_date` to calculate timestamp
            save_date["eid"] = save_date["eid"].astype(int)

            # ukb['eid'] = ukb["eid"].astype(int)
            ukb = ukb.merge(
                save_date[["eid", "start_timestamp"]], on="eid", how="inner"
            )
            ukb["start_timestamp"] = pd.to_datetime(ukb["start_timestamp"])
            ukb["date_time"] = ukb["start_timestamp"] + pd.to_timedelta(
                (ukb["freq"] - 1) * 60, unit="s"
            )
            ukb["date"] = ukb["date_time"].dt.date
            ukb["hour"] = ukb["date_time"].dt.hour
            ukb["minute_temp"] = ukb["date_time"].dt.minute
            ukb["minute"] = ukb["minute_temp"].apply(lambda x: f"{x:02d}")
            ukb["time"] = ukb["hour"].astype(str) + ":" + ukb["minute"]

            # Calculate day
            day = ukb[["eid", "date"]].drop_duplicates().copy()
            day["day"] = day.groupby("eid").cumcount() + 1

            # Add day information back to `ukb`
            ukb = ukb.merge(day, on=["eid", "date"], how="left")

            # Append the processed DataFrame to the list
            data = pd.concat([data, ukb])

    data = data[["date_time", "enmo_mg"]]
    data.rename(
        columns={"enmo_mg": "enmo", "date_time": "timestamp"}, inplace=True
    )
    data.set_index("timestamp", inplace=True)
    data.sort_index(inplace=True)

    # rescale ENMO to g - should be /1000 however value range suggests that /100 is better to make it comparable with other sources
    data["enmo"] = data["enmo"]

    # only keep ENMO >= 0.1 else 0
    data["enmo"] = data["enmo"].apply(lambda x: x if x >= 0.1 else 0)

    meta_dict["raw_n_datapoints"] = data.shape[0]
    meta_dict["raw_start_datetime"] = data.index.min()
    meta_dict["raw_end_datetime"] = data.index.max()
    meta_dict["raw_data_frequency"] = "minute-level"
    meta_dict["raw_data_type"] = "ENMO"
    meta_dict["raw_data_unit"] = "mg"

    if verbose:
        print(
            f"Loaded {data.shape[0]} minute-level ENMO records from {enmo_file_dir}"
        )

    return data[["enmo"]]


def filter_ukb_data(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Filter UK Biobank accelerometer data to ensure data quality.

    This function applies data quality filters to UK Biobank ENMO data, including
    removal of incomplete days and selection of the longest consecutive sequence.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ENMO data with datetime index and 'ENMO' column.
    meta_dict : dict, default={}
        Dictionary to store metadata about the filtering process.
    verbose : bool, default=False
        Whether to print filtering information and progress.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only complete and consecutive days of data.
        Maintains same structure as input DataFrame.

    Notes
    -----
    - Removes incomplete days using filter_incomplete_days (requires 1440 points per day)
    - Selects longest consecutive sequence using filter_consecutive_days
    - Assumes minute-level data (1/60 Hz sampling frequency)
    - Updates metadata with information about the filtering process

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample UK Biobank data
    >>> dates = pd.date_range('2023-01-01', periods=10000, freq='min')
    >>> data = pd.DataFrame({'ENMO': np.random.uniform(0, 0.1, 10000)}, index=dates)
    >>>
    >>> # Filter the data
    >>> meta_dict = {}
    >>> filtered_data = filter_ukb_data(data, meta_dict=meta_dict, verbose=True)
    >>> print(f"Original data points: {len(data)}")
    >>> print(f"Filtered data points: {len(filtered_data)}")
    """
    _data = data.copy()

    _data = filter_incomplete_days(
        _data, data_freq=1 / 60, expected_points_per_day=1440
    )
    if verbose:
        print(
            f"Filtered out {data.shape[0] - _data.shape[0]} minute-level ENMO records due to incomplete daily coverage"
        )

    n_old = _data.shape[0]
    _data = filter_consecutive_days(_data)
    if verbose:
        print(
            f"Filtered out {n_old - _data.shape[0]} minute-level ENMO records due to non-consecutive days"
        )

    return _data


def resample_ukb_data(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Resample UK Biobank accelerometer data to ensure consistent 1-minute intervals.

    This function ensures consistent minute-level resolution for UK Biobank ENMO data
    by resampling to 1-minute intervals and handling any gaps in the data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ENMO data with datetime index and 'ENMO' column.
    meta_dict : dict, default={}
        Dictionary to store metadata about the resampling process.
    verbose : bool, default=False
        Whether to print resampling information and progress.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with consistent 1-minute intervals.
        Missing values are interpolated linearly and any remaining gaps are
        filled using backward fill.

    Notes
    -----
    - Uses pandas resample('1min') with linear interpolation
    - Applies backward fill (bfill) to handle any remaining gaps
    - Ensures consistent temporal resolution for analysis
    - Maintains data integrity and structure

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample UK Biobank data with irregular intervals
    >>> dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:30',
    ...                         '2023-01-01 00:03:00', '2023-01-01 00:04:30'])
    >>> data = pd.DataFrame({
    : [0.1, 0.2, 0.3, 0.4]}, index=dates)
    >>>
    >>> # Resample to minute level
    >>> meta_dict = {}
    >>> resampled_data = resample_ukb_data(data, meta_dict=meta_dict, verbose=True)
    >>> print(f"Original data points: {len(data)}")
    >>> print(f"Resampled data points: {len(resampled_data)}")
    """
    _data = data.copy()

    _data = _data.resample("1min").mean().interpolate(method="linear").bfill()
    if verbose:
        print(f"Resampled {data.shape[0]} to {_data.shape[0]} timestamps")

    return _data
