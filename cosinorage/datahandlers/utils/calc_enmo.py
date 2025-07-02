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

import numpy as np
import pandas as pd


def calculate_enmo(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Calculate the Euclidean Norm Minus One (ENMO) metric from accelerometer data.

    This function computes the ENMO metric, which is a widely used measure in physical
    activity research for quantifying acceleration while accounting for gravity.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing accelerometer data with columns:
        - 'x': X-axis acceleration values
        - 'y': Y-axis acceleration values
        - 'z': Z-axis acceleration values
        All values should be in g units (1g = 9.81 m/s²).
    verbose : bool, default=False
        If True, prints processing information.

    Returns
    -------
    numpy.ndarray
        Array of ENMO values. Values are truncated at 0, meaning negative
        values are set to 0. Returns np.nan if calculation fails.

    Notes
    -----
    - ENMO = sqrt(x² + y² + z²) - 1
    - Values are truncated at 0 (negative values become 0)
    - ENMO represents acceleration in excess of 1g (gravity)
    - Commonly used in physical activity and sleep research
    - Handles errors gracefully by returning np.nan

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample accelerometer data
    >>> data = pd.DataFrame({
    ...     'x': [0.1, 0.2, 0.3],
    ...     'y': [0.1, 0.2, 0.3],
    ...     'z': [1.0, 1.1, 1.2]  # Close to 1g (gravity)
    ... })
    >>>
    >>> # Calculate ENMO
    >>> enmo_values = calculate_enmo(data, verbose=True)
    >>> print(f"ENMO values: {enmo_values}")
    >>> # Output: [0.014, 0.028, 0.042] (approximately)
    """

    if data.empty:
        return pd.DataFrame()

    try:
        _acc_vectors = data[["x", "y", "z"]].values
        _enmo_vals = np.linalg.norm(_acc_vectors, axis=1) - 1
        _enmo_vals = np.maximum(_enmo_vals, 0)
    except Exception as e:
        print(f"Error calculating ENMO: {e}")
        _enmo_vals = np.nan

    if verbose:
        print(f"Calculated ENMO for {data.shape[0]} accelerometer records")

    return _enmo_vals


def calculate_minute_level_enmo(
    data: pd.DataFrame, meta_dict: dict = {}, verbose: bool = False
) -> pd.DataFrame:
    """
    Resample high-frequency ENMO data to minute-level by averaging over each minute.

    This function aggregates high-frequency ENMO data to minute-level resolution
    using mean aggregation, which is the standard approach for circadian rhythm analysis.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and 'ENMO' column containing high-frequency ENMO data.
        Optional 'wear' column for wear time information.
    meta_dict : dict, default={}
        Dictionary containing metadata. Should include:
        - 'sf': Sampling frequency in Hz (defaults to 25Hz if not specified)
    verbose : bool, default=False
        If True, prints processing information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing minute-level aggregated data with:
        - 'ENMO': Mean ENMO value for each minute
        - 'wear': Mean wear time for each minute (if wear column exists in input)
        Index is datetime at minute resolution.

    Raises
    ------
    ValueError
        If sampling frequency is less than 1/60 Hz (less than one sample per minute).

    Notes
    -----
    - Uses pandas resample('min').mean() for aggregation
    - Handles both ENMO and wear columns if present
    - Converts index to datetime format
    - Standard preprocessing step for circadian rhythm analysis
    - Handles errors gracefully by returning empty DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> # Create sample high-frequency ENMO data
    >>> dates = pd.date_range('2023-01-01 00:00:00', periods=3600, freq='S')  # 1 hour of second-level data
    >>> data = pd.DataFrame({
    ...     'ENMO': np.random.uniform(0, 0.1, 3600),
    ...     'wear': np.random.choice([0, 1], 3600)
    ... }, index=dates)
    >>>
    >>> # Resample to minute level
    >>> meta_dict = {'sf': 1}  # 1 Hz sampling frequency
    >>> minute_data = calculate_minute_level_enmo(data, meta_dict=meta_dict, verbose=True)
    >>> print(f"Original records: {len(data)}")
    >>> print(f"Minute-level records: {len(minute_data)}")
    """

    # Get sampling frequency from meta_dict or use default
    sf = meta_dict.get("sf", 25)  # Default to 25Hz if not specified

    if sf < 1 / 60:
        raise ValueError("Sampling frequency must be at least 1 minute")

    if data.empty:
        return pd.DataFrame()

    try:
        minute_level_enmo_df = (
            data["enmo"].resample("min").mean().to_frame(name="enmo")
        )
        # check if data has a wear column
        if "wear" in data.columns:
            minute_level_enmo_df["wear"] = data["wear"].resample("min").mean()

    except Exception as e:
        print(f"Error resampling ENMO data: {e}")
        minute_level_enmo_df = pd.DataFrame()

    minute_level_enmo_df.index = pd.to_datetime(minute_level_enmo_df.index)

    if verbose:
        print(
            f"Aggregated ENMO values at the minute level leading to {minute_level_enmo_df.shape[0]} records"
        )

    return minute_level_enmo_df
