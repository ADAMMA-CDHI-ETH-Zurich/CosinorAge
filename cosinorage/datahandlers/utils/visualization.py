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

import matplotlib.pyplot as plt
from scipy.signal import welch
from tqdm import tqdm


def plot_orig_enmo(
    acc_handler, resample: str = "15min", wear: bool = True
) -> None:
    """
    Plot the original ENMO values resampled at a specified interval.

    This function creates a time series plot of ENMO (Euclidean Norm Minus One) values
    with optional highlighting of wear and non-wear periods. The data is resampled
    to reduce noise and improve visualization clarity.

    Parameters
    ----------
    acc_handler : DataHandler
        Accelerometer data handler object containing the raw data. Must have:
        - get_sf_data(): Method returning DataFrame with 'ENMO' and 'wear' columns
    resample : str, default='15min'
        The resampling interval for the plot. Can be any pandas time frequency string
        (e.g., '5min', '1H', '1D').
    wear : bool, default=True
        Whether to add color bands for wear and non-wear periods.
        - True: Shows red bands for non-wear periods
        - False: Shows only the ENMO time series

    Returns
    -------
    None
        Displays a matplotlib plot.

    Notes
    -----
    - The function resamples the data using mean aggregation
    - Non-wear periods are highlighted with red bands when wear=True
    - The plot uses a progress bar (tqdm) when processing wear data
    - The figure size is set to 12x6 inches

    Examples
    --------
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Load data
    >>> handler = GenericDataHandler('data.csv')
    >>>
    >>> # Plot with wear periods highlighted
    >>> plot_orig_enmo(handler, resample='30min', wear=True)
    >>>
    >>> # Plot without wear highlighting
    >>> plot_orig_enmo(handler, resample='1H', wear=False)
    """
    _data = (
        acc_handler.get_sf_data()
        .resample(f"{resample}")
        .mean()
        .reset_index(inplace=False)
    )

    plt.figure(figsize=(12, 6))
    plt.plot(_data["timestamp"], _data["enmo"], label="ENMO", color="black")

    if wear:
        # Add color bands for wear and non-wear periods
        # add tqdm progress bar

        for i in tqdm(range(len(_data) - 1)):
            if _data["wear"].iloc[i] != 1:
                start_time = _data["timestamp"].iloc[i]
                end_time = _data["timestamp"].iloc[i + 1]
                color = "red"
                plt.axvspan(start_time, end_time, color=color, alpha=0.3)

    plt.show()


def plot_enmo(handler) -> None:
    """
    Plot minute-level ENMO values with optional wear/non-wear period highlighting.

    This function creates a time series plot of minute-level ENMO values with
    automatic highlighting of wear and non-wear periods using colored bands.

    Parameters
    ----------
    handler : DataHandler
        Data handler object containing the minute-level ENMO data. Must have:
        - get_ml_data(): Method returning DataFrame with 'ENMO' column
        - Optional 'wear' column for wear/non-wear periods

    Returns
    -------
    None
        Displays a matplotlib plot showing ENMO values over time with optional
        wear/non-wear period highlighting in green/red.

    Notes
    -----
    - Wear periods are highlighted in green
    - Non-wear periods are highlighted in red
    - The plot automatically adjusts Y-axis limits to show the full range
    - If no 'wear' column is present, only the ENMO time series is shown
    - The figure size is set to 12x6 inches

    Examples
    --------
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Load data
    >>> handler = GenericDataHandler('data.csv')
    >>>
    >>> # Plot minute-level ENMO with wear highlighting
    >>> plot_enmo(handler)
    """
    _data = handler.get_ml_data().reset_index(inplace=False)
    
    if "index" in _data.columns:
        _data.rename(columns={"index": "timestamp"}, inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(_data["timestamp"], _data["enmo"], label="ENMO", color="black")

    if "wear" in _data.columns and _data["wear"].max() != -1:
        plt.fill_between(
            _data["timestamp"],
            _data["wear"] * max(_data["enmo"]) * 1.25,
            color="green",
            alpha=0.5,
            label="wear",
        )
        plt.fill_between(
            _data["timestamp"],
            (1 - _data["wear"]) * max(_data["enmo"]) * 1.25,
            color="red",
            alpha=0.5,
            label="non-wear",
        )
        plt.legend()

    plt.ylim(0, max(_data["enmo"]) * 1.25)
    plt.show()


def plot_orig_enmo_freq(acc_handler) -> None:
    """
    Plot the frequency domain representation of the original ENMO signal using Welch's method.

    This function computes and displays the power spectral density (PSD) of the ENMO signal
    using Welch's method, which provides a smoothed estimate of the signal's frequency content.

    Parameters
    ----------
    acc_handler : DataHandler
        Accelerometer data handler object containing the raw ENMO data. Must have:
        - get_sf_data(): Method returning DataFrame with 'ENMO' column

    Returns
    -------
    None
        Displays a matplotlib plot showing the power spectral density of the ENMO signal
        computed using Welch's method.

    Notes
    -----
    - Uses scipy.signal.welch for power spectral density estimation
    - Sampling frequency is set to 80 Hz
    - Segment length is set to 1024 samples for frequency resolution
    - The plot shows frequency (Hz) on the x-axis and power spectral density on the y-axis
    - The figure size is set to 20x5 inches

    Examples
    --------
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Load data
    >>> handler = GenericDataHandler('data.csv')
    >>>
    >>> # Plot frequency domain representation
    >>> plot_orig_enmo_freq(handler)
    """
    # convert to frequency domain
    f, Pxx = welch(acc_handler.get_sf_data()["enmo"], fs=80, nperseg=1024)

    plt.figure(figsize=(20, 5))
    plt.plot(f, Pxx)
    plt.show()
