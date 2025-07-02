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
from scipy.signal import butter, filtfilt


def remove_noise(
    data: pd.DataFrame,
    sf: float,
    filter_type: str = "lowpass",
    filter_cutoff: float = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Remove noise from accelerometer data using a Butterworth filter.

    This function applies a digital Butterworth filter to remove noise from
    accelerometer data. The filter can be configured as lowpass, highpass,
    bandpass, or bandstop depending on the noise characteristics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing accelerometer data with columns ['x', 'y', 'z'].
        Data should have a datetime index and contain acceleration values in g units.
    sf : float
        Sampling frequency of the accelerometer data in Hz.
    filter_type : str, default='lowpass'
        Type of filter to apply. Must be one of:
        - 'lowpass': Removes high-frequency noise above cutoff
        - 'highpass': Removes low-frequency noise below cutoff
        - 'bandpass': Keeps frequencies between two cutoff values
        - 'bandstop': Removes frequencies between two cutoff values
    filter_cutoff : float or list, default=2
        Cutoff frequency(ies) for the filter in Hz.
        - For lowpass/highpass: single float value
        - For bandpass/bandstop: list of two values [low_cutoff, high_cutoff]
    verbose : bool, default=False
        Whether to print progress information during filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with noise removed from the ['x', 'y', 'z'] columns.
        The filtered data maintains the same structure as the input.

    Raises
    ------
    ValueError
        If filter_type is 'bandpass' or 'bandstop' but filter_cutoff is not a list
        of two values.
        If filter_type is 'lowpass' or 'highpass' but filter_cutoff is not a single
        numeric value.
        If the input DataFrame is empty.
    KeyError
        If the DataFrame does not contain required columns ['x', 'y', 'z'].

    Notes
    -----
    - Uses scipy.signal.butter and scipy.signal.filtfilt for zero-phase filtering
    - The filter order is fixed at 2 (second-order Butterworth filter)
    - The function applies the same filter to all three axes (x, y, z)
    - Zero-phase filtering is used to avoid phase distortion

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample accelerometer data with noise
    >>> timestamps = pd.date_range('2023-01-01', periods=1000, freq='40ms')
    >>> data = pd.DataFrame({
    ...     'x': np.random.normal(0, 0.1, 1000) + 0.5*np.sin(2*np.pi*0.1*np.arange(1000)),
    ...     'y': np.random.normal(0, 0.1, 1000) + 0.3*np.cos(2*np.pi*0.05*np.arange(1000)),
    ...     'z': np.random.normal(1, 0.1, 1000)  # Gravity component
    ... }, index=timestamps)
    >>>
    >>> # Remove high-frequency noise with lowpass filter
    >>> filtered_data = remove_noise(data, sf=25, filter_type='lowpass',
    ...                              filter_cutoff=2, verbose=True)
    >>>
    >>> # Remove low-frequency drift with highpass filter
    >>> filtered_data = remove_noise(data, sf=25, filter_type='highpass',
    ...                              filter_cutoff=0.1, verbose=True)
    """
    if (filter_type == "bandpass" or filter_type == "bandstop") and (
        type(filter_cutoff) != list or len(filter_cutoff) != 2
    ):
        raise ValueError(
            "Bandpass and bandstop filters require a list of two cutoff frequencies."
        )

    if (filter_type == "highpass" or filter_type == "lowpass") and type(
        filter_cutoff
    ) not in [float, int]:
        raise ValueError(
            "Highpass and lowpass filters require a single cutoff frequency."
        )

    if data.empty:
        raise ValueError("Dataframe is empty.")

    if not all(col in data.columns for col in ["x", "y", "z"]):
        raise KeyError("Dataframe must contain 'x', 'y' and 'z' columns.")

    def butter_lowpass_filter(data, cutoff, sf, btype, order=2):
        # Design Butterworth filter
        nyquist = 0.5 * sf  # Nyquist frequency
        normal_cutoff = np.array(cutoff) / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)

        # Apply filter to data
        return filtfilt(b, a, data)

    _data = data.copy()

    cutoff = filter_cutoff
    _data["x"] = butter_lowpass_filter(
        _data["x"], cutoff, sf, btype=filter_type
    )
    _data["y"] = butter_lowpass_filter(
        _data["y"], cutoff, sf, btype=filter_type
    )
    _data["z"] = butter_lowpass_filter(
        _data["z"], cutoff, sf, btype=filter_type
    )

    if verbose:
        print("Noise removal done")

    return _data[["x", "y", "z"]]
