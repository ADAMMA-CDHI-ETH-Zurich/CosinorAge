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
from skdh.sleep.sleep_classification import compute_sleep_predictions
from skdh.sleep.endpoints import WakeAfterSleepOnset, TotalSleepTime, PercentTimeAsleep, NumberWakeBouts, SleepOnsetLatency

from typing import List
from itertools import tee


def apply_sleep_wake_predictions(data: pd.DataFrame, sleep_params: dict) -> pd.DataFrame:
    """
    Apply sleep-wake prediction to a DataFrame with ENMO values.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing ENMO values in a column named 'ENMO'.

    Returns
    -------
    pd.Series
        Series containing sleep predictions where:
        1 = sleep
        0 = wake

    Raises
    ------
    ValueError
        If 'ENMO' column is not found in DataFrame.
    """
    if "ENMO" not in data.columns:
        raise ValueError(f"Column ENMO not found in the DataFrame.")
    
    data_ = data.copy()
    # make sf higher
    sf = sleep_params.get("sleep_ck_sf", 0.0025)
    rescore = sleep_params.get("sleep_rescore", True)

    result = compute_sleep_predictions(data_["ENMO"], sf=sf, rescore=rescore)
    data_['sleep'] = pd.DataFrame(result, columns=['sleep']).set_index(data_.index)['sleep']

    return data_['sleep']

def WASO(data: pd.DataFrame) -> List[int]:
    """
    Calculate Wake After Sleep Onset (WASO) for each 24-hour cycle.

    WASO represents the total time spent awake after the first sleep onset 
    until the final wake time.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (0=sleep, 1=wake)

    Returns
    -------
    pd.Series
        Series indexed by date containing WASO values in minutes for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    Zero is returned for days where no sleep is detected.
    """
    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    waso = []
    w = WakeAfterSleepOnset()

    # Group by 24-hour cycle
    for date, day_data in daily_groups:
        # Sort by timestamp within the group
        day_data = day_data.sort_index()

        waso.append(int(w.predict(np.array(day_data["sleep"]))))

    return waso

def TST(data: pd.DataFrame) -> List[int]:
    """
    Calculate Total Sleep Time (TST) for each 24-hour cycle.

    TST represents the total time spent in sleep state during the analysis period.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    pd.Series
        Series indexed by date containing total sleep time in minutes for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    Sleep time is calculated by counting all epochs marked as sleep (0).
    """

    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    tst = []
    t = TotalSleepTime()

    for date, day_data in daily_groups:
        day_data = day_data.sort_index()

        tst.append(int(t.predict(np.array(day_data["sleep"]))))
    
    return tst

def PTA(data: pd.DataFrame) -> List[float]:
    """
    Calculate Percent Time Asleep (PTA) for each 24-hour cycle.

    PTA represents the percentage of time spent asleep relative to the total recording time.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    pd.Series
        Series indexed by date containing percent time asleep (0-1) for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    PTA is calculated as: (number of sleep epochs) / (total number of epochs).
    """
    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    pta = []
    p = PercentTimeAsleep()

    for date, day_data in daily_groups:
        # Sort by timestamp within the group
        day_data = day_data.sort_index()

        pta.append(float(p.predict(np.array(day_data["sleep"]))))
    
    return pta

def NWB(data: pd.DataFrame) -> List[int]:
    """
    Calculate Number of Wake Bouts (NWB) for each 24-hour cycle.

    NWB represents the count of distinct wake episodes occurring between sleep periods
    during the analysis period.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    List[int]
        List containing the number of wake bouts for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    A wake bout is defined as a continuous period of wake states between
    two sleep states.
    """

    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    nwb = []
    n = NumberWakeBouts()

    for date, day_data in daily_groups:
        day_data = day_data.sort_index()

        nwb.append(int(n.predict(np.array(day_data["sleep"]))))
    
    return nwb

def SOL(data: pd.DataFrame) -> List[int]:
    """
    Calculate Sleep Onset Latency (SOL) for each 24-hour cycle.

    SOL represents the time taken to fall asleep, measured from the start of the
    recording period until the first sleep onset.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)

    Returns
    -------
    List[int]
        List containing sleep onset latency in minutes for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    SOL is calculated as the time from the start of the recording until
    the first detected sleep episode.
    """
    data_ = data.copy()

    daily_groups = data_.groupby(data_.index.date)
    sol = []
    s = SleepOnsetLatency()

    for date, day_data in daily_groups:
        day_data = day_data.sort_index()

        sol.append(int(s.predict(np.array(day_data["sleep"]))))
    
    return sol

def SRI(data: pd.DataFrame) -> float:
    """
    Calculate Sleep Regularity Index (SRI) for each 24-hour cycle.

    SRI quantifies the day-to-day similarity of sleep-wake patterns. It ranges from -100 
    (completely irregular) to +100 (perfectly regular).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (1=sleep, 0=wake)
        Must contain at least 2 complete days of data.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date containing SRI values for each day (starting from day 2).
        The SRI column contains values ranging from -100 to +100.

    Raises
    ------
    ValueError
        If less than 2 complete days of data are provided.

    Notes
    -----
    - SRI is calculated by comparing sleep states between consecutive 24-hour periods
    - The first day will not have an SRI value as it requires a previous day for comparison
    - Incomplete days at the end of the recording are trimmed
    - Formula: SRI = (2 * concordance_rate - 1) * 100
    """

    if data.empty:
        return np.nan
    
    data_ = data.copy()
    data_ = data_.sort_index()

    N = 1440  # minutes per day
    M = len(pd.unique(data_.index.date))

    if M < 2:  # Need at least 2 days for SRI
        return np.nan
    
    daily_groups = data_.groupby(data_.index.date)
    sri = 0

    def overlapping_pairs(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    for (date_prev, day_data_prev), (date_next, day_data_next) in overlapping_pairs(daily_groups):
        # check for concordance of sleep states between consecutive days
        concordance = (day_data_prev["sleep"].reset_index(drop=True) == day_data_next["sleep"].reset_index(drop=True)).sum()
        sri += concordance 

    sri = float(-100 + 200/(M*(N-1)) * sri)
    
    return sri
