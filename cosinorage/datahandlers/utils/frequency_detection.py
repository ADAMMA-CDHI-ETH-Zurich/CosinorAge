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
import numpy as np
from collections import Counter

def detect_frequency_from_timestamps(timestamps):
    """
    Detect sampling frequency by finding the most common time delta.
    
    Args:
        timestamps: Series or array of datetime objects
    
    Returns:
        float: Sampling frequency in Hz
    """
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, errors='coerce')
    timestamps = pd.Series(timestamps).dropna()
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required to detect frequency.")

    # Calculate all time deltas
    time_deltas = timestamps.diff().dropna()
    # Convert to seconds
    if hasattr(time_deltas, 'dt'):
        time_deltas_seconds = time_deltas.dt.total_seconds()
    else:
        # If already timedelta64[ns] dtype, convert directly
        time_deltas_seconds = time_deltas.astype('timedelta64[s]').astype(float)
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
        raise ValueError("Most common time delta is zero, cannot determine frequency.")
    # Calculate frequency
    frequency = 1.0 / most_common_delta
    return frequency