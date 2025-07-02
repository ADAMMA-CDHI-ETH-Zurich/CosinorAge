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
from skdh.preprocessing import AccelThresholdWearDetection


def detect_wear_periods(
    data: pd.DataFrame,
    sf: float,
    sd_crit: float,
    range_crit: float,
    window_length: int,
    window_skip: int,
    meta_dict: dict = {},
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Detect periods of device wear using acceleration thresholds.

    This function identifies when the accelerometer device is being worn by analyzing
    the standard deviation and range of acceleration data within sliding windows.
    The algorithm is based on the assumption that worn devices show more variable
    acceleration patterns than unworn devices.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed accelerometer data with datetime index and columns ['x', 'y', 'z'].
        Data should be in g units and cleaned of major artifacts.
    sf : float
        Sampling frequency of the accelerometer data in Hz.
    sd_crit : float
        Standard deviation criterion for wear detection. Threshold for the minimum
        standard deviation required to classify a window as "worn".
    range_crit : float
        Range criterion for wear detection. Threshold for the minimum range of
        acceleration values required to classify a window as "worn".
    window_length : int
        Length of the sliding window in seconds. Longer windows provide more
        stable wear detection but may miss brief wear periods.
    window_skip : int
        Number of seconds to skip between consecutive windows. Controls the
        temporal resolution of wear detection.
    meta_dict : dict, default={}
        Dictionary to store wear detection metadata and parameters.
    verbose : bool, default=False
        Whether to print progress information during wear detection.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary wear detection column ['wear'] where:
        - 1 indicates the device is being worn
        - 0 indicates the device is not being worn
        The DataFrame has the same index as the input data.

    Notes
    -----
    - Uses skdh.preprocessing.AccelThresholdWearDetection for the core algorithm
    - The function converts acceleration data from g to mg units for processing
    - Wear periods are determined by analyzing both standard deviation and range
    - The algorithm is sensitive to the choice of sd_crit and range_crit parameters

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample accelerometer data
    >>> timestamps = pd.date_range('2023-01-01', periods=1000, freq='40ms')
    >>> data = pd.DataFrame({
    ...     'x': np.random.normal(0, 0.1, 1000),
    ...     'y': np.random.normal(0, 0.1, 1000),
    ...     'z': np.random.normal(1, 0.1, 1000)  # Gravity component
    ... }, index=timestamps)
    >>>
    >>> # Detect wear periods
    >>> wear_data = detect_wear_periods(
    ...     data, sf=25, sd_crit=0.013, range_crit=0.05,
    ...     window_length=60, window_skip=30, verbose=True
    ... )
    >>> print(f"Wear time: {wear_data['wear'].sum() / 25:.1f} seconds")
    """
    _data = data.copy()

    time = np.array(_data.index.astype("int64") // 10**9)
    acc = np.array(_data[["x", "y", "z"]]).astype(np.float64) / 1000

    # wear_predictor = CountWearDetection()
    wear_predictor = AccelThresholdWearDetection(
        sd_crit=sd_crit,
        range_crit=range_crit,
        window_length=window_length,
        window_skip=window_skip,
    )
    ranges = wear_predictor.predict(time=time, accel=acc, fs=sf)["wear"]

    wear_array = np.zeros(len(data.index))
    for start, end in ranges:
        wear_array[start : end + 1] = 1

    _data["wear"] = pd.DataFrame(wear_array, columns=["wear"]).set_index(
        data.index
    )

    if verbose:
        print("Wear detection done")

    return _data[["wear"]]


def calc_weartime(
    data: pd.DataFrame, sf: float, meta_dict: dict, verbose: bool
) -> None:
    """
    Calculate total, wear, and non-wear time from accelerometer data.

    This function computes summary statistics about device wear time based on
    wear detection results. It calculates the total recording duration, time
    the device was worn, and time the device was not worn.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing accelerometer data with a 'wear' column indicating
        wear status (1 for worn, 0 for not worn). Should have a datetime index.
    sf : float
        Sampling frequency of the accelerometer data in Hz.
    meta_dict : dict
        Dictionary to store wear time metadata. Will be updated with the following keys:
        - 'total_time': Total recording time in seconds
        - 'wear_time': Time device was worn in seconds
        - 'non-wear_time': Time device was not worn in seconds
    verbose : bool
        Whether to print progress information during calculation.

    Returns
    -------
    None
        Updates meta_dict with wear time statistics.

    Notes
    -----
    - Total time is calculated from the first to last timestamp
    - Wear time is calculated by summing the 'wear' column and converting to seconds
    - Non-wear time is calculated as total_time - wear_time
    - All times are stored in seconds in the meta_dict

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data with wear detection
    >>> timestamps = pd.date_range('2023-01-01', periods=1000, freq='40ms')
    >>> data = pd.DataFrame({
    ...     'wear': np.random.choice([0, 1], 1000, p=[0.3, 0.7])  # 70% wear time
    ... }, index=timestamps)
    >>>
    >>> # Calculate wear time statistics
    >>> meta_dict = {}
    >>> calc_weartime(data, sf=25, meta_dict=meta_dict, verbose=True)
    >>> print(f"Total time: {meta_dict['total_time']:.1f} seconds")
    >>> print(f"Wear time: {meta_dict['wear_time']:.1f} seconds")
    >>> print(f"Non-wear time: {meta_dict['non-wear_time']:.1f} seconds")
    """
    _data = data.copy()

    total = float((_data.index[-1] - _data.index[0]).total_seconds())
    wear = float((_data["wear"].sum()) * (1 / sf))
    nonwear = float((total - wear))

    meta_dict.update(
        {"total_time": total, "wear_time": wear, "non-wear_time": nonwear}
    )
    if verbose:
        print("Wear time calculated")
