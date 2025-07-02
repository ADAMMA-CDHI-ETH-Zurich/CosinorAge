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


def activity_metrics(
    data: pd.Series, pa_params: dict = cutpoints
) -> pd.DataFrame:
    r"""Calculate Sedentary Behavior (SB), Light Physical Activity (LIPA), and
    Moderate-to-Vigorous Physical Activity (MVPA) durations in minutes for each day.

    This function classifies physical activity levels based on ENMO (Euclidean Norm Minus One)
    values using established cutpoints and returns the duration spent in each activity level
    for each day in the dataset.

    Parameters
    ----------
    data : pd.Series
        A pandas Series with a DatetimeIndex and ENMO (Euclidean Norm Minus One) values.
        The index should be datetime with minute-level resolution.
        The values should be float numbers representing acceleration in g units.
    pa_params : dict, default=cutpoints
        Dictionary containing physical activity cutpoints:
        - 'sl' or 'pa_cutpoint_sl': Sedentary behavior threshold (default: 0.030g)
        - 'lm' or 'pa_cutpoint_lm': Light activity threshold (default: 0.100g)
        - 'mv' or 'pa_cutpoint_mv': Moderate-to-vigorous activity threshold (default: 0.400g)

    Returns
    -------
    tuple
        Tuple containing four lists of daily activity durations in minutes:
        - sedentary_minutes: Minutes spent in sedentary behavior (ENMO ≤ sl)
        - light_minutes: Minutes spent in light physical activity (sl < ENMO ≤ lm)
        - moderate_minutes: Minutes spent in moderate physical activity (lm < ENMO ≤ mv)
        - vigorous_minutes: Minutes spent in vigorous physical activity (ENMO > mv)

    Raises
    ------
    ValueError
        If required cutpoints are not found in the pa_params dictionary.

    Notes
    -----
    - The function assumes minute-level data and returns durations in minutes
    - ENMO cutpoints are based on established thresholds for physical activity classification
    - Cutpoints may vary depending on accelerometer type, position, and user characteristics
    - Returns empty lists if input data is empty
    - Groups data by date and calculates daily totals

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample ENMO data for one day
    >>> dates = pd.date_range('2023-01-01', periods=1440, freq='min')  # One day
    >>> enmo_data = pd.Series(np.random.uniform(0, 0.5, 1440), index=dates)
    >>>
    >>> # Calculate activity metrics
    >>> sb, lipa, mvpa, vig = activity_metrics(enmo_data)
    >>> print(f"Sedentary minutes: {sb[0]}")
    >>> print(f"Light activity minutes: {lipa[0]}")
    >>> print(f"Moderate activity minutes: {mvpa[0]}")
    >>> print(f"Vigorous activity minutes: {vig[0]}")
    >>>
    >>> # Use custom cutpoints
    >>> custom_cutpoints = {
    ...     'pa_cutpoint_sl': 0.020,
    ...     'pa_cutpoint_lm': 0.080,
    ...     'pa_cutpoint_mv': 0.300
    ... }
    >>> sb, lipa, mvpa, vig = activity_metrics(enmo_data, pa_params=custom_cutpoints)
    """

    if data.empty:
        return [], [], [], []

    data_ = data.copy()[["enmo"]]

    if "sl" not in cutpoints and "pa_cutpoint_sl" not in cutpoints:
        raise ValueError(
            "Sedentary cutpoint not found in cutpoints dictionary"
        )
    if "lm" not in cutpoints and "pa_cutpoint_lm" not in cutpoints:
        raise ValueError("Light cutpoint not found in cutpoints dictionary")
    if "mv" not in cutpoints and "pa_cutpoint_mv" not in cutpoints:
        raise ValueError(
            "Moderate-to-Vigorous cutpoint not found in cutpoints dictionary"
        )

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
        sedentary_minutes.append(int((day_data["enmo"] <= sl).sum()))
        light_minutes.append(
            int(((day_data["enmo"] > sl) & (day_data["enmo"] <= lm)).sum())
        )
        moderate_minutes.append(
            int(((day_data["enmo"] > lm) & (day_data["enmo"] <= mv)).sum())
        )
        vigorous_minutes.append(int((day_data["enmo"] > mv).sum()))

    return sedentary_minutes, light_minutes, moderate_minutes, vigorous_minutes
