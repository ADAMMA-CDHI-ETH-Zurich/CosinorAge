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
from tqdm import tqdm
from scipy.signal import welch

def plot_orig_enmo(acc_handler, resample: str = '15min', wear: bool = True):
    """
    Plot the original ENMO values resampled at a specified interval.

    Args:
        acc_handler: Accelerometer data handler object containing the raw data
        resample (str): The resampling interval (default is '15min')
        wear (bool): Whether to add color bands for wear and non-wear periods (default is True)

    Returns:
        None: Displays a matplotlib plot
    """
    #_data = self.acc_df.resample('5min').mean().reset_index(inplace=False)
    _data = acc_handler.get_sf_data().resample(f'{resample}').mean().reset_index(inplace=False)
    

    plt.figure(figsize=(12, 6))
    plt.plot(_data['TIMESTAMP'], _data['ENMO'], label='ENMO', color='black')

    if wear:
        # Add color bands for wear and non-wear periods
        # add tqdm progress bar

        for i in tqdm(range(len(_data) - 1)):
            if _data['wear'].iloc[i] != 1:
                start_time = _data['TIMESTAMP'].iloc[i]
                end_time = _data['TIMESTAMP'].iloc[i + 1]
                color = 'red'
                plt.axvspan(start_time, end_time, color=color, alpha=0.3)

    plt.show()

def plot_enmo(handler):
    """
    Plot minute-level ENMO values with optional wear/non-wear period highlighting.

    Args:
        handler: Data handler object containing the minute-level ENMO data

    Returns:
        None: Displays a matplotlib plot showing ENMO values over time with optional
            wear/non-wear period highlighting in green/red
    """
    _data = handler.get_ml_data().reset_index(inplace=False)

    plt.figure(figsize=(12, 6))
    plt.plot(_data['TIMESTAMP'], _data['ENMO'], label='ENMO', color='black')

    if 'wear' in _data.columns and _data['wear'].max() != -1:
        plt.fill_between(_data['TIMESTAMP'], _data['wear']*max(_data['ENMO'])*1.25, color='green', alpha=0.5, label='wear')
        plt.fill_between(_data['TIMESTAMP'], (1-_data['wear'])*max(_data['ENMO'])*1.25, color='red', alpha=0.5, label='non-wear')
        plt.legend()
        
    plt.ylim(0, max(_data['ENMO'])*1.25)
    plt.show()

def plot_orig_enmo_freq(acc_handler):
    """
    Plot the frequency domain representation of the original ENMO signal using Welch's method.

    Args:
        acc_handler: Accelerometer data handler object containing the raw ENMO data

    Returns:
        None: Displays a matplotlib plot showing the power spectral density of the ENMO signal
            computed using Welch's method with a sampling frequency of 80Hz and segment length of 1024
    """
    # convert to frequency domain
    f, Pxx = welch(acc_handler.get_sf_data()['ENMO'], fs=80, nperseg=1024)

    plt.figure(figsize=(20, 5))
    plt.plot(f, Pxx)
    plt.show()