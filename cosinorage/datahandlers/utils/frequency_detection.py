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


def detect_frequency_from_timestamps(timestamps: pd.Series) -> float:
    """
    Detect sampling frequency by finding the most common time delta.

    This function analyzes a series of timestamps to determine the sampling frequency
    of the data by calculating the time differences between consecutive samples and
    finding the most frequently occurring interval.

    Parameters
    ----------
    timestamps : pd.Series
        Series or array of datetime objects representing the timestamps of data points.
        Can be pandas datetime objects, numpy datetime64, or string timestamps that
        can be converted to datetime.

    Returns
    -------
    float
        Sampling frequency in Hz (samples per second).

    Raises
    ------
    ValueError
        If less than two timestamps are provided.
        If no time deltas can be calculated.
        If the most common time delta is zero.
        If the mode cannot be determined.

    Notes
    -----
    - The function converts all timestamps to pandas datetime format
    - Time deltas are calculated in seconds
    - The most common (mode) time delta is used to determine frequency
    - Frequency is calculated as 1.0 / most_common_delta

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Regular 25 Hz sampling
    >>> timestamps = pd.date_range('2023-01-01', periods=100, freq='40ms')
    >>> freq = detect_frequency_from_timestamps(timestamps)
    >>> print(f"Detected frequency: {freq:.1f} Hz")
    Detected frequency: 25.0 Hz
    >>>
    >>> # Irregular sampling with some missing points
    >>> irregular_times = pd.to_datetime([
    ...     '2023-01-01 00:00:00',
    ...     '2023-01-01 00:00:00.040',
    ...     '2023-01-01 00:00:00.080',
    ...     '2023-01-01 00:00:00.120',
    ...     '2023-01-01 00:00:00.200',  # Gap here
    ...     '2023-01-01 00:00:00.240'
    ... ])
    >>> freq = detect_frequency_from_timestamps(irregular_times)
    >>> print(f"Detected frequency: {freq:.1f} Hz")
    Detected frequency: 25.0 Hz
    """
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, errors="coerce")
    timestamps = pd.Series(timestamps).dropna()
    if len(timestamps) < 2:
        raise ValueError(
            "At least two timestamps are required to detect frequency."
        )

    # Calculate all time deltas in ms
    time_deltas = timestamps.diff().dropna()
    # Convert to seconds
    if hasattr(time_deltas, "dt"):
        time_deltas_seconds = time_deltas.dt.total_seconds()
    else:
        # If already timedelta64[ns] dtype, convert directly
        time_deltas_seconds = time_deltas.astype("timedelta64[s]").astype(
            float
        )
    # Convert to pandas Series to use mode()
    time_deltas_series = pd.Series(time_deltas_seconds)
    # Find the most common delta (majority)
    if time_deltas_series.empty:
        raise ValueError("Not enough time deltas to determine frequency.")
    mode = time_deltas_series.mode()
    if mode.empty:
        raise ValueError("Could not determine the most common time delta.")
    most_common_delta = mode.iloc[0]
    if most_common_delta == 0:
        raise ValueError(
            "Most common time delta is zero, cannot determine frequency."
        )
    # Calculate frequency
    frequency = 1.0 / most_common_delta
    return frequency
