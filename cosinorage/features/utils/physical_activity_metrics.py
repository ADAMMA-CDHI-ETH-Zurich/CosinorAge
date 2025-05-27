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

# cutpoint heavily depend on the accelerometer used, the position of the accelerometer and the user (gender, age, ...)
cutpoints = {
    "sl": 0.030,
    "lm": 0.100, 
    "mv": 0.400,
}


def activity_metrics(data: pd.Series, pa_params: dict = cutpoints) -> pd.DataFrame:
    r"""Calculate Sedentary Behavior (SB), Light Physical Activity (LIPA), and 
    Moderate-to-Vigorous Physical Activity (MVPA) durations in hours for each day.

    Parameters
    ----------
    data : pd.Series
        A pandas Series with a DatetimeIndex and ENMO (Euclidean Norm Minus One) values.
        The index should be datetime with minute-level resolution.
        The values should be float numbers representing acceleration in g units.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily physical activity metrics:
        - Index: date (datetime.date)
        - Columns:
            - SB: Hours spent in sedentary behavior (ENMO ≤ 0.00001g)
            - LIPA: Hours spent in light physical activity (0.00001g < ENMO ≤ 0.01g)
            - MVPA: Hours spent in moderate-to-vigorous physical activity (ENMO > 0.01g)

    Notes
    -----
    - The function assumes minute-level data when converting to hours
    - ENMO cutpoints are based on established thresholds:
        - SB: ≤ 0.00001g
        - LIPA: > 0.00001g and ≤ 0.01g
        - MVPA: > 0.01g
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')  # One day
    >>> enmo = pd.Series(np.random.uniform(0, 0.1, 1440), index=dates)
    >>> activity_metrics(enmo)
              SB      LIPA      MVPA
    2023-01-01  8.2  10.5  5.3
    """

    if data.empty:
        return [], [], [], []

    data_ = data.copy()[['ENMO']]

    if "sl" not in cutpoints and "pa_cutpoint_sl" not in cutpoints:
        raise ValueError("Sedentary cutpoint not found in cutpoints dictionary")
    if "lm" not in cutpoints and "pa_cutpoint_lm" not in cutpoints:
        raise ValueError("Light cutpoint not found in cutpoints dictionary")
    if "mv" not in cutpoints and "pa_cutpoint_mv" not in cutpoints:
        raise ValueError("Moderate-to-Vigorous cutpoint not found in cutpoints dictionary")

    # Group data by day
    daily_groups = data_.groupby(data_.index.date)

    # Initialize list to store results
    sedentary_minutes = []
    light_minutes = []
    moderate_minutes = []
    vigorous_minutes = []

    # if not in dict, take "sl"
    sl = pa_params.get("pa_cutpoint_sl", cutpoints.get("sl"))
    lm = pa_params.get("pa_cutpoint_lm", cutpoints.get("lm"))
    mv = pa_params.get("pa_cutpoint_mv", cutpoints.get("mv"))
    
    for date, day_data in daily_groups:
        sedentary_minutes.append(int((day_data['ENMO'] <= sl).sum()))
        light_minutes.append(int(((day_data['ENMO'] > sl) & (day_data['ENMO'] <= lm)).sum()))
        moderate_minutes.append(int(((day_data['ENMO'] > lm) & (day_data['ENMO'] <= mv)).sum()))
        vigorous_minutes.append(int((day_data['ENMO'] > mv).sum()))
    

    return sedentary_minutes, light_minutes, moderate_minutes, vigorous_minutes