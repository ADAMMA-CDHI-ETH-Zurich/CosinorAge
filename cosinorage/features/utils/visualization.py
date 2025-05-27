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

def plot_sleep_predictions(feature_obj, simple=True, start_date=None, end_date=None):
    """Plot sleep predictions over time.
    
    Creates visualization of sleep/wake predictions, optionally including non-wear periods.
    Simple mode shows a binary plot with dots, while detailed mode shows ENMO data
    with colored bands for sleep/wake states.

    Args:
        feature_obj: Feature object containing ml_data with sleep predictions
        simple (bool, optional): If True, shows simple binary plot. If False, shows detailed plot. Defaults to True.
        start_date (datetime, optional): Start date for plotting. Defaults to None (earliest date).
        end_date (datetime, optional): End date for plotting. Defaults to None (latest date).

    Returns:
        None: Displays the plot using matplotlib
    """
    if start_date is None:
        start_date = feature_obj.ml_data.index[0]
    if end_date is None:
        end_date = feature_obj.ml_data.index[-1]
    selected_data = feature_obj.ml_data[(feature_obj.ml_data.index >= start_date) & (feature_obj.ml_data.index <= end_date)]
    if simple:
        plt.figure(figsize=(30, 0.5))
        plt.plot(selected_data["sleep"] == 0, 'g.', label='Wake')
        plt.plot(selected_data["sleep"] != 0, 'b.', label='Sleep')
        if 'wear' in selected_data.columns:
            plt.plot(selected_data["wear"] == 0, 'r.', label='Non-wear')
        plt.ylim(0.9, 1.1)
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        max_y = max(selected_data['ENMO'])*1.25
        plt.figure(figsize=(30, 6))
        # plot sleep predictions as red bands
        plt.fill_between(selected_data.index, (1-selected_data['sleep'])*max_y, color='green', alpha=0.5, label='Wake')
        plt.fill_between(selected_data.index, selected_data['sleep']*max_y, color='blue', alpha=0.5, label='Sleep')
        if 'wear' in selected_data.columns:
            plt.fill_between(selected_data.index, (1-selected_data['wear'])*max_y, color='red', alpha=0.5, label='Non-wear')
        plt.plot(selected_data['ENMO'], label='ENMO', color='black')
        # y axis limits
        plt.ylim(0, max_y)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("ENMO")
        plt.show()

def plot_non_wear(feature_obj, simple=True, start_date=None, end_date=None):
    """Plot non-wear periods over time.
    
    Creates visualization of wear/non-wear periods. Simple mode shows a binary plot
    with dots, while detailed mode shows ENMO data with colored bands for wear states.

    Args:
        feature_obj: Feature object containing ml_data with wear/non-wear predictions
        simple (bool, optional): If True, shows simple binary plot. If False, shows detailed plot. Defaults to True.
        start_date (datetime, optional): Start date for plotting. Defaults to None (earliest date).
        end_date (datetime, optional): End date for plotting. Defaults to None (latest date).

    Returns:
        None: Displays the plot using matplotlib
    """
    if start_date is None:
        start_date = feature_obj.ml_data.index[0]
    if end_date is None:
        end_date = feature_obj.ml_data.index[-1]
    selected_data = feature_obj.ml_data[(feature_obj.ml_data.index >= start_date) & (feature_obj.ml_data.index <= end_date)]
    if simple:
        plt.figure(figsize=(20, 0.5))
        plt.plot(selected_data["wear"] == 1, 'g.', label='Wear')
        plt.plot(selected_data["wear"] == 0, 'r.', label='Non-wear')
        plt.ylim(0.9, 1.1)
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(30, 6))
        plt.plot(selected_data['ENMO'], label='ENMO', color='black')
        # plot sleep predictions as red bands
        plt.fill_between(selected_data.index, (1-selected_data['wear'])*1000, color='red', alpha=0.5, label='Non-wear')
        plt.fill_between(selected_data.index, selected_data['wear']*1000, color='green', alpha=0.5, label='Wear')
        # y axis limits
        plt.ylim(0, max(selected_data['ENMO'])*1.25)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("ENMO")
        plt.show()

def plot_cosinor(feature_obj):
    """Plot cosinor analysis results for activity rhythm analysis.
    
    Creates detailed visualizations of circadian rhythm analysis showing raw activity data (ENMO)
    overlaid with fitted cosinor curves. Includes markers for key circadian parameters:
    MESOR (rhythm-adjusted mean), amplitude, and acrophase (peak timing).

    Args:
        feature_obj: Feature object containing cosinor analysis results and ENMO data
        multiday (bool, optional): If True, shows analysis across all days combined. 
            If False, shows individual daily plots. Defaults to True.

    Returns:
        None: Displays the plot(s) using matplotlib

    Raises:
        ValueError: If cosinor features haven't been computed (either multiday or by-day)
    """
    if "cosinor_fitted" not in feature_obj.ml_data.columns:
        raise ValueError("Cosinor fitted values not computed.")
    minutes = np.arange(0, len(feature_obj.ml_data))
    timestamps = feature_obj.ml_data.index
    plt.figure(figsize=(20, 10))
    plt.plot(timestamps, feature_obj.ml_data["ENMO"], 'r-')
    plt.plot(timestamps, feature_obj.ml_data["cosinor_fitted"], 'b-')
    plt.ylim(0, max(feature_obj.ml_data["ENMO"])*1.5)
    cosinor_keys = ["mesor", "amplitude", "acrophase", "acrophase_time"]
    if all(key in feature_obj.feature_dict['cosinor'].keys() for key in cosinor_keys):
        # x ticks should be daytime hours
        plt.axhline(feature_obj.feature_dict['cosinor']["mesor"], color='green', linestyle='--', label='MESOR')
    plt.show()