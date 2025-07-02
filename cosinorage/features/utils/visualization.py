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
import numpy as np


def plot_sleep_predictions(
    feature_obj, simple=True, start_date=None, end_date=None
):
    """
    Plot sleep predictions over time.

    Creates visualization of sleep/wake predictions, optionally including non-wear periods.
    Simple mode shows a binary plot with dots, while detailed mode shows ENMO data
    with colored bands for sleep/wake states.

    Parameters
    ----------
    feature_obj : WearableFeatures
        Feature object containing ml_data with sleep predictions. Must have:
        - ml_data: DataFrame with 'sleep' column (1=sleep, 0=wake)
        - Optional 'wear' column for non-wear periods
        - 'ENMO' column for detailed plotting
    simple : bool, default=True
        If True, shows simple binary plot with dots for sleep/wake states.
        If False, shows detailed plot with ENMO data and colored bands.
    start_date : datetime, optional
        Start date for plotting. If None, uses the earliest date in the data.
    end_date : datetime, optional
        End date for plotting. If None, uses the latest date in the data.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    - Simple mode: Shows binary sleep/wake states as colored dots
    - Detailed mode: Shows ENMO activity data with colored bands for sleep/wake
    - Non-wear periods are shown in red if 'wear' column is available
    - The function automatically handles date range selection

    Examples
    --------
    >>> from cosinorage.features import WearableFeatures
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Load data and compute features
    >>> handler = GenericDataHandler('data.csv')
    >>> features = WearableFeatures(handler)
    >>>
    >>> # Plot sleep predictions
    >>> plot_sleep_predictions(features, simple=True)
    >>> plot_sleep_predictions(features, simple=False,
    ...                       start_date='2023-01-01', end_date='2023-01-02')
    """
    if start_date is None:
        start_date = feature_obj.ml_data.index[0]
    if end_date is None:
        end_date = feature_obj.ml_data.index[-1]
    selected_data = feature_obj.ml_data[
        (feature_obj.ml_data.index >= start_date)
        & (feature_obj.ml_data.index <= end_date)
    ]
    if simple:
        plt.figure(figsize=(30, 0.5))
        plt.plot(selected_data["sleep"] == 0, "g.", label="Wake")
        plt.plot(selected_data["sleep"] != 0, "b.", label="Sleep")
        if "wear" in selected_data.columns:
            plt.plot(selected_data["wear"] == 0, "r.", label="Non-wear")
        plt.ylim(0.9, 1.1)
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        max_y = max(selected_data["enmo"]) * 1.25
        plt.figure(figsize=(30, 6))
        # plot sleep predictions as red bands
        plt.fill_between(
            selected_data.index,
            (1 - selected_data["sleep"]) * max_y,
            color="green",
            alpha=0.5,
            label="Wake",
        )
        plt.fill_between(
            selected_data.index,
            selected_data["sleep"] * max_y,
            color="blue",
            alpha=0.5,
            label="Sleep",
        )
        if "wear" in selected_data.columns:
            plt.fill_between(
                selected_data.index,
                (1 - selected_data["wear"]) * max_y,
                color="red",
                alpha=0.5,
                label="Non-wear",
            )
        plt.plot(selected_data["enmo"], label="ENMO", color="black")
        # y axis limits
        plt.ylim(0, max_y)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("ENMO")
        plt.show()


def plot_non_wear(feature_obj, simple=True, start_date=None, end_date=None):
    """
    Plot non-wear periods over time.

    Creates visualization of wear/non-wear periods. Simple mode shows a binary plot
    with dots, while detailed mode shows ENMO data with colored bands for wear states.

    Parameters
    ----------
    feature_obj : WearableFeatures
        Feature object containing ml_data with wear/non-wear predictions. Must have:
        - ml_data: DataFrame with 'wear' column (1=worn, 0=not worn)
        - 'ENMO' column for detailed plotting
    simple : bool, default=True
        If True, shows simple binary plot with dots for wear/non-wear states.
        If False, shows detailed plot with ENMO data and colored bands.
    start_date : datetime, optional
        Start date for plotting. If None, uses the earliest date in the data.
    end_date : datetime, optional
        End date for plotting. If None, uses the latest date in the data.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    - Simple mode: Shows binary wear/non-wear states as colored dots
    - Detailed mode: Shows ENMO activity data with colored bands for wear states
    - Non-wear periods are highlighted in red, wear periods in green
    - The function automatically handles date range selection

    Examples
    --------
    >>> from cosinorage.features import WearableFeatures
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Load data and compute features
    >>> handler = GenericDataHandler('data.csv')
    >>> features = WearableFeatures(handler)
    >>>
    >>> # Plot wear/non-wear periods
    >>> plot_non_wear(features, simple=True)
    >>> plot_non_wear(features, simple=False,
    ...               start_date='2023-01-01', end_date='2023-01-02')
    """
    if start_date is None:
        start_date = feature_obj.ml_data.index[0]
    if end_date is None:
        end_date = feature_obj.ml_data.index[-1]
    selected_data = feature_obj.ml_data[
        (feature_obj.ml_data.index >= start_date)
        & (feature_obj.ml_data.index <= end_date)
    ]
    if simple:
        plt.figure(figsize=(20, 0.5))
        plt.plot(selected_data["wear"] == 1, "g.", label="Wear")
        plt.plot(selected_data["wear"] == 0, "r.", label="Non-wear")
        plt.ylim(0.9, 1.1)
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(30, 6))
        plt.plot(selected_data["enmo"], label="ENMO", color="black")
        # plot sleep predictions as red bands
        plt.fill_between(
            selected_data.index,
            (1 - selected_data["wear"]) * 1000,
            color="red",
            alpha=0.5,
            label="Non-wear",
        )
        plt.fill_between(
            selected_data.index,
            selected_data["wear"] * 1000,
            color="green",
            alpha=0.5,
            label="Wear",
        )
        # y axis limits
        plt.ylim(0, max(selected_data["enmo"]) * 1.25)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("ENMO")
        plt.show()


def plot_cosinor(feature_obj):
    """
    Plot cosinor analysis results for activity rhythm analysis.

    Creates detailed visualizations of circadian rhythm analysis showing raw activity data (ENMO)
    overlaid with fitted cosinor curves. Includes markers for key circadian parameters:
    MESOR (rhythm-adjusted mean), amplitude, and acrophase (peak timing).

    Parameters
    ----------
    feature_obj : WearableFeatures
        Feature object containing cosinor analysis results and ENMO data.
        The ml_data DataFrame must contain 'ENMO' and 'cosinor_fitted' columns.
        The feature_dict must contain a 'cosinor' key with mesor, amplitude,
        acrophase, and acrophase_time values.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Raises
    ------
    ValueError
        If cosinor features haven't been computed (missing 'cosinor_fitted' column).

    Notes
    -----
    - Shows raw ENMO data in red and fitted cosinor curve in blue
    - MESOR is displayed as a horizontal green dashed line
    - The plot provides visual validation of the cosinor fit quality
    - Y-axis limits are automatically adjusted to show the full range of data

    Examples
    --------
    >>> from cosinorage.features import WearableFeatures
    >>> from cosinorage.datahandlers import GenericDataHandler
    >>>
    >>> # Load data and compute features
    >>> handler = GenericDataHandler('data.csv')
    >>> features = WearableFeatures(handler)
    >>>
    >>> # Plot cosinor analysis results
    >>> plot_cosinor(features)
    """
    if "cosinor_fitted" not in feature_obj.ml_data.columns:
        raise ValueError("Cosinor fitted values not computed.")
    np.arange(0, len(feature_obj.ml_data))
    timestamps = feature_obj.ml_data.index
    plt.figure(figsize=(20, 10))
    plt.plot(timestamps, feature_obj.ml_data["enmo"], "r-")
    plt.plot(timestamps, feature_obj.ml_data["cosinor_fitted"], "b-")
    plt.ylim(0, max(feature_obj.ml_data["enmo"]) * 1.5)
    cosinor_keys = ["mesor", "amplitude", "acrophase", "acrophase_time"]
    if all(
        key in feature_obj.feature_dict["cosinor"].keys()
        for key in cosinor_keys
    ):
        # x ticks should be daytime hours
        plt.axhline(
            feature_obj.feature_dict["cosinor"]["mesor"],
            color="green",
            linestyle="--",
            label="MESOR",
        )
    plt.show()
