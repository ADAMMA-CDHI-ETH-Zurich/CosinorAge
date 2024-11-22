import pandas as pd
import numpy as np

def apply_sleep_wake_predictions(df: pd.DataFrame, mode: str="sleeppy") -> pd.DataFrame:
    """
    Apply sleep-wake prediction to a DataFrame with ENMO values.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing ENMO values.
    enmo_column (str): Column name containing ENMO values.

    Returns:
    pd.DataFrame: DataFrame with an additional 'sleep_predictions' column.
    """
    if "ENMO" not in df.columns:
        raise ValueError(f"Column ENMO not found in the DataFrame.")
    
    df_ = df.copy()

    if mode == "sleeppy":
        # Run sleep-wake predictions using ColeKripke
        ck = ColeKripke(df_["ENMO"])  # Assuming ColeKripke class is implemented
        df_["sleep_predictions"] = ck.predict(sf=1)  # Add predictions to the DataFrame
    elif mode == "ggir":
        df_["sleep_predictions"] = enmo_sleep_wake_windows(df_)
    else:
        raise ValueError(f"Mode {mode} not supported.")
    
    return df_["sleep_predictions"]

def waso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WASO (Wake After Sleep Onset) for a 24-hour cycle (12 PM to 12 PM).

    Parameters:
    df (pd.DataFrame): DataFrame with timestamp as index and sleep-wake predictions.

    Returns:
    pd.DataFrame: DataFrame with WASO values for each 24-hour cycle.
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
            first_sleep_idx = group[group["sleep_predictions"] == 0].index[0]  # First occurrence of sleep
        except IndexError:
            # No sleep detected in this cycle
            waso_results.append({"day": date, "waso_minutes": 0})
            continue
        
        # Calculate WASO: sum wake states (1) after first sleep onset
        waso = group.loc[first_sleep_idx:, "sleep_predictions"].sum()
        time_interval_minutes = (group.index[1] - group.index[0]).seconds / 60.0
        
        waso_results.append({
            "day": date,
            "waso_minutes": float(waso * time_interval_minutes)
        })
    
    return pd.DataFrame(waso_results).set_index("day")["waso_minutes"]

def tst(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total sleep time for a 24-hour cycle (12 PM to 12 PM).
    """

    df_ = df.copy()

    df_.index = pd.to_datetime(df_.index)
    df_['day'] = df_.index.date  # Extract date

    sleep_results = []

    for date, group in df_.groupby('day'):
        # Sort by timestamp within the group
        group = group.sort_index()

        # Calculate total sleep time: sum sleep states (0)
        total_sleep = group[group["sleep_predictions"] == 0].shape[0]
        sleep_results.append({
            "day": date,
            "total_sleep_minutes": total_sleep
        })
    
    return pd.DataFrame(sleep_results).set_index("day")["total_sleep_minutes"]

def pta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percent time asleep for a 24-hour cycle (12 PM to 12 PM).
    """
    df_ = df.copy()
    df_.index = pd.to_datetime(df_.index)
    df_['day'] = df_.index.date  # Extract date

    sleep_results = []

    for date, group in df_.groupby('day'):
        # Sort by timestamp within the group
        group = group.sort_index()

        # Calculate percent time asleep: sum sleep states (0) / total states
        percent_time_asleep = group[group["sleep_predictions"] == 0].shape[0] / group.shape[0]
        sleep_results.append({
            "day": date,
            "percent_time_asleep": percent_time_asleep
        })
    
    return pd.DataFrame(sleep_results).set_index("day")["percent_time_asleep"]

def sri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sleep regularity for a 24-hour cycle (12 PM to 12 PM).
    """

    sleep_states = df["sleep_predictions"].values
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

class ColeKripke:
    """
    Runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and activity is represented
    by an activity index.
    """

    def __init__(self, activity_index):
        """
        Initialization of the class

        :param activity_index: pandas dataframe of epoch level activity index values
        """
        self.activity_index = activity_index
        self.predictions = None

    def predict(self, sf:float=0.1):
        """
        Runs the prediction of sleep wake states based on activity index data.

        :param sf: scale factor to use for the predictions (default corresponds to scale factor optimized for use with
        the activity index, if other activity measures are desired the scale factor can be modified or optimized.)
        The recommended range for the scale factor is between 0.1 and 0.25 depending on the sensitivity to activity
        desired, and possibly the population being observed.

        :return: rescored predictions
        """
        kernel = (
            sf
            * np.array([4.64, 6.87, 3.75, 5.07, 16.19, 5.84, 4.024, 0.00, 0.00])[::-1]
        )
        scores = np.convolve(self.activity_index, kernel, "same")
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        # rescore the original predictions
        for i in range(3):
            self.rescore(scores)
        self.predictions = scores
        return self.predictions

    def rescore(self, predictions):
        """
        Application of Webster's rescoring rules as described in the Cole-Kripke paper.

        :param predictions: array of predictions
        :return: rescored predictions
        """
        rescored = predictions.copy()
        # rules a through c
        wake_bin = 0
        for t in range(len(rescored)):
            if rescored[t] == 1:
                wake_bin += 1
            else:
                if (
                    14 < wake_bin
                ):  # rule c: at least 15 minutes of wake, next 4 minutes of sleep get rescored
                    rescored[t : t + 4] = 1.0
                elif (
                    9 < wake_bin < 15
                ):  # rule b: at least 10 minutes of wake, next 3 minutes of sleep get rescored
                    rescored[t : t + 3] = 1.0
                elif (
                    3 < wake_bin < 10
                ):  # rule a: at least 4 minutes of wake, next 1 minute of sleep gets rescored
                    rescored[t] = 1.0
                wake_bin = 0
        # rule d: 6 minutes or less of sleep surrounded by at least 10 minutes of wake on each side gets rescored
        sleep_bin = 0
        start_ind = 0
        for t in range(10, len(rescored) - 10):
            if rescored[t] == 0:
                sleep_bin += 1
                if sleep_bin == 1:
                    start_ind = t
            else:
                if 0 < sleep_bin <= 6:
                    if (
                        sum(rescored[start_ind - 10 : start_ind]) == 10.0
                        and sum(rescored[t : t + 10]) == 10.0
                    ):
                        rescored[start_ind:t] = 1.0
                sleep_bin = 0
        self.predictions = rescored

def enmo_sleep_wake_windows(df: pd.DataFrame, threshold: float=0.03, epoch_size: int=60, min_sleep_duration: int=60) -> pd.DataFrame:
    """
    Classifies a time series into sleep and wake periods based on ENMO values.

    Parameters:
    - ts (pd.DataFrame): Time series DataFrame with a 'time' column.
    - enmo (pd.Series): ENMO values corresponding to the time series.
    - threshold (float): ENMO threshold to classify sleep (default: 10 mg).
    - epoch_size (int): Duration of each epoch in seconds (default: 60 seconds).
    - min_sleep_duration (int): Minimum duration for a valid sleep period in minutes.

    Returns:
    - pd.DataFrame: Updated time series with a 'diur' column (1: sleep, 0: wake).
    """

    if "ENMO" not in df.columns:
        raise ValueError(f"Column ENMO not found in the DataFrame.")

    df_ = df.copy()
    df_['sleep_predictions'] = 0  # Initialize diurnal classification (0: wake)

    # Identify potential sleep epochs (ENMO < threshold)
    is_sleep = df_["ENMO"] < threshold

    # Detect consecutive sleep periods using run-length encoding (RLE)
    sleep_periods = []
    start_idx = None
    for i, sleep in enumerate(is_sleep):
        if sleep:
            if start_idx is None:
                start_idx = i  # Start of a sleep period
        else:
            if start_idx is not None:
                duration = (i - start_idx) * (epoch_size / 60)  # Convert to minutes
                if duration >= min_sleep_duration:
                    sleep_periods.append((start_idx, i - 1))  # Save valid sleep period
                start_idx = None

    # Handle final sleep period if it extends to the end
    if start_idx is not None:
        duration = (len(is_sleep) - start_idx) * (epoch_size / 60)
        if duration >= min_sleep_duration:
            sleep_periods.append((start_idx, len(is_sleep) - 1))

    # Mark sleep periods in the 'diur' column
    for start, end in sleep_periods:
        df_.iloc[start:end, df_.columns.get_loc('sleep_predictions')] = 1

    return 1 - df_["sleep_predictions"]