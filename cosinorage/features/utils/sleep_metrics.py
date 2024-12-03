import pandas as pd
import numpy as np
from skdh.sleep.sleep_classification import compute_sleep_predictions


def apply_sleep_wake_predictions(data: pd.DataFrame) -> pd.DataFrame:
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
        0 = sleep
        1 = wake

    Raises
    ------
    ValueError
        If 'ENMO' column is not found in DataFrame.
    """
    if "ENMO" not in data.columns:
        raise ValueError(f"Column ENMO not found in the DataFrame.")
    
    data_ = data.copy()
    # make sf higher
    result = compute_sleep_predictions(data_["ENMO"], sf=0.075)
    data_['sleep'] = pd.DataFrame(result, columns=['sleep']).set_index(data_.index)['sleep']

    return data_['sleep']

def waso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Wake After Sleep Onset (WASO) for each 24-hour cycle.

    WASO represents the total time spent awake after the first sleep onset 
    until the final wake time.

    Parameters
    ----------
    df : pd.DataFrame
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
    df_ = df.copy()

    # Ensure the index is in datetime format
    df_.index = pd.to_datetime(df_.index)
    
    # Assign each record to a 24-hour cycle starting at 12 PM
    df_['day'] = df_.index.date  # Extract date

    waso_results = []

    # Group by 24-hour cycle
    for date, group in df_.groupby('day'):
        # Sort by timestamp within the group
        group = group.sort_index()
        
        # Identify sleep onset (first transition from wake to sleep)
        try:
            first_sleep_idx = group[group["sleep"] == 0].index[0]  # First occurrence of sleep
        except IndexError:
            # No sleep detected in this cycle
            waso_results.append({"day": date, "waso_minutes": 0})
            continue
        
        # Calculate WASO: sum wake states (1) after first sleep onset
        waso = group.loc[first_sleep_idx:, "sleep"].sum()
        time_interval_minutes = (group.index[1] - group.index[0]).seconds / 60.0
        
        waso_results.append({
            "day": date,
            "waso_minutes": float(waso * time_interval_minutes)
        })
    
    return pd.DataFrame(waso_results).set_index("day")["waso_minutes"]

def tst(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Total Sleep Time (TST) for each 24-hour cycle.

    TST represents the total time spent in sleep state during the analysis period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (0=sleep, 1=wake)

    Returns
    -------
    pd.Series
        Series indexed by date containing total sleep time in minutes for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    Sleep time is calculated by counting all epochs marked as sleep (0).
    """

    df_ = df.copy()

    df_.index = pd.to_datetime(df_.index)
    df_['day'] = df_.index.date  # Extract date

    sleep_results = []

    for date, group in df_.groupby('day'):
        # Sort by timestamp within the group
        group = group.sort_index()

        # Calculate total sleep time: sum sleep states (0)
        total_sleep = group[group["sleep"] == 0].shape[0]
        sleep_results.append({
            "day": date,
            "total_sleep_minutes": total_sleep
        })
    
    return pd.DataFrame(sleep_results).set_index("day")["total_sleep_minutes"]

def pta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Percent Time Asleep (PTA) for each 24-hour cycle.

    PTA represents the percentage of time spent asleep relative to the total recording time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (0=sleep, 1=wake)

    Returns
    -------
    pd.Series
        Series indexed by date containing percent time asleep (0-1) for each 24-hour cycle.

    Notes
    -----
    The function processes data in 24-hour cycles starting at midnight.
    PTA is calculated as: (number of sleep epochs) / (total number of epochs).
    """
    df_ = df.copy()
    df_.index = pd.to_datetime(df_.index)
    df_['day'] = df_.index.date  # Extract date

    sleep_results = []

    for date, group in df_.groupby('day'):
        # Sort by timestamp within the group
        group = group.sort_index()

        # Calculate percent time asleep: sum sleep states (0) / total states
        percent_time_asleep = group[group["sleep"] == 0].shape[0] / group.shape[0]
        sleep_results.append({
            "day": date,
            "percent_time_asleep": percent_time_asleep
        })
    
    return pd.DataFrame(sleep_results).set_index("day")["percent_time_asleep"]

def sri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Sleep Regularity Index (SRI) for each 24-hour cycle.

    SRI quantifies the day-to-day similarity of sleep-wake patterns. It ranges from -100 
    (completely irregular) to +100 (perfectly regular).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'sleep' column (0=sleep, 1=wake)
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

    sleep_states = df["sleep"].values
    epochs_per_day = 24 * 60 
    epochs_per_window = epochs_per_day * 2

    if len(sleep_states) < 2 * epochs_per_day:
        raise ValueError("Insufficient data. At least two complete days are required.")
    # Remove extra epochs to ensure an integer number of days

    total_epochs = len(sleep_states)
    extra_epochs = total_epochs % epochs_per_day
    if extra_epochs > 0:
        sleep_states = sleep_states[:-extra_epochs]

    sri_results = []
    for start in range(epochs_per_day, len(sleep_states), epochs_per_day):
        # Extract current and previous day's data
        prev_day = sleep_states[start - epochs_per_day : start]
        curr_day = sleep_states[start : start + epochs_per_day]

        # Compare sleep states between the two days
        concordance = prev_day == curr_day
        concordance_rate = np.mean(concordance)

        # Calculate SRI
        sri = float(200 * concordance_rate - 100)
        sri_results.append(sri)

    # Build the output DataFrame
    dates = df.index[epochs_per_day::epochs_per_day]  # Start at second day's date
    sri_df = pd.DataFrame({"date": dates, "SRI": sri_results}).set_index("date") # NaN for the first day

    return sri_df
