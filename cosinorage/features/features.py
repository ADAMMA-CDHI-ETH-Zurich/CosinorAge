import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..dataloaders import DataLoader
from .utils.nonparam_analysis import *
from .utils.physical_activity_metrics import *
from .utils.sleep_metrics import *
from .utils.cosinor_analysis import *

class WearableFeatures:
    """A class for computing and managing features from wearable accelerometer data.

    This class processes raw ENMO (Euclidean Norm Minus One) data to compute various
    circadian rhythm and physical activity metrics, including cosinor analysis,
    non-parametric measures, activity levels, and sleep metrics.

    Attributes:
        enmo (pd.DataFrame): Raw ENMO data with datetime index
        feature_df (pd.DataFrame): Computed features with date index
        feature_dict (dict): Additional feature storage (if needed)
    """

    def __init__(self, loader: DataLoader):
        """Initialize WearableFeatures with data from a DataLoader.

        Args:
            loader (DataLoader): DataLoader instance containing ENMO data
        """
        self.enmo = loader.get_enmo_data().copy()
        self.feature_df = pd.DataFrame(index=pd.unique(self.enmo.index.date))
        self.feature_dict = {}   

    def run(self):
        """Compute all available features at once."""
        self.compute_cosinor_features()
        self.compute_IV()
        self.compute_IS()
        self.compute_RA()
        self.compute_M10()
        self.compute_L5()
        self.compute_SB()
        self.compute_LIPA()
        self.compute_MVPA()
        self.compute_sleep_predictions()
        self.compute_TST()
        self.compute_WASO()
        self.compute_PTA()
        self.compute_SRI()

    def compute_cosinor_features(self):
        """Compute cosinor analysis features including MESOR, amplitude, and acrophase.
        
        Updates feature_df with:
            - MESOR: Midline Estimating Statistic Of Rhythm
            - amplitude: Half the peak-to-trough difference
            - acrophase: Peak time of the rhythm in radians
            - acrophase_time: Peak time of the rhythm in minutes from midnight
        """
        cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]

        # by day cosinor computation
        if not all(col in self.feature_df.columns for col in cosinor_columns) or "cosinor_fitted" not in self.enmo.columns:
            params, fitted = cosinor_by_day(self.enmo)
            self.feature_df["MESOR"] = params["MESOR"]
            self.feature_df["amplitude"] = params["amplitude"]
            self.feature_df["acrophase"] = params["acrophase"]
            self.feature_df["acrophase_time"] = params["acrophase_time"]
            self.enmo["cosinor_by_day_fitted"] = fitted

        # multiday cosinor computation
        if not all(col in self.feature_dict for col in cosinor_columns):
            params, fitted = cosinor_multiday(self.enmo)
            self.feature_dict.update(params)
            self.enmo["cosinor_multiday_fitted"] = fitted

    def get_cosinor_features(self):
        """Get computed cosinor features.
        
        Returns:
            pd.DataFrame: DataFrame containing MESOR, amplitude, acrophase, and acrophase_time
        """
        cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]
        if not all(col in self.feature_df.columns for col in cosinor_columns):
            self.compute_cosinor_features()

        return pd.DataFrame(self.feature_df[cosinor_columns]), {key: self.feature_dict[key] for key in cosinor_columns}
        
    def compute_IV(self):
        """Compute Intradaily Variability (IV).
        
        IV quantifies the fragmentation of the rhythm, with higher values
        indicating more frequent transitions between rest and activity.
        """
        if "IV" not in self.feature_df.columns:
            self.feature_df["IV"] = IV(self.enmo)

    def get_IV(self):
        """Get computed Intradaily Variability (IV) values.
        
        Returns:
            pd.DataFrame: DataFrame containing IV values representing the 
                         fragmentation of rest-activity patterns
        """
        if "IV" not in self.feature_df.columns:
            self.compute_IV()
        return pd.DataFrame(self.feature_df["IV"])

    def compute_IS(self):
        """Compute Interdaily Stability (IS).
        
        IS quantifies the stability of rhythms between days, with higher values
        indicating more consistent day-to-day patterns.
        """
        if "IS" not in self.feature_df.columns:
            self.feature_df["IS"] = IS(self.enmo)

    def get_IS(self):
        """Get computed Interdaily Stability (IS) values.
        
        Returns:
            pd.DataFrame: DataFrame containing IS values
        """
        if "IS" not in self.feature_df.columns:
            self.compute_IS()
        return pd.DataFrame(self.feature_df["IS"])

    def compute_RA(self):
        """Compute Relative Amplitude (RA).
        
        RA is calculated as (M10 - L5)/(M10 + L5), where M10 is the most active
        10-hour period and L5 is the least active 5-hour period.
        """
        if "RA" not in self.feature_df.columns:
            self.feature_df["RA"] = RA(self.enmo)

    def get_RA(self):
        """Get computed Relative Amplitude (RA) values.
        
        Returns:
            pd.DataFrame: DataFrame containing RA values representing the 
                         relative difference between most and least active periods
        """
        if "RA" not in self.feature_df.columns:
            self.compute_RA()
        return pd.DataFrame(self.feature_df["RA"])

    def compute_M10(self):
        """Compute M10 (Most active 10-hour period).
        
        Updates feature_df with:
            - M10: Activity level during most active 10-hour period
            - M10_start: Start time of most active period
        """
        if "M10" or "M10_start" not in self.feature_df.columns:
            res = M10(self.enmo)
            self.feature_df["M10"] = res["M10"]
            self.feature_df["M10_start"] = res["M10_start"]

    def get_M10(self):
        """Get computed M10 (Most active 10-hour period) values.
        
        Returns:
            pd.DataFrame: DataFrame containing M10 values representing activity 
                         levels during the most active 10-hour period of each day
        """
        if "M10" not in self.feature_df.columns:
            self.compute_M10()
        return pd.DataFrame(self.feature_df["M10"])

    def compute_L5(self):
        """Compute L5 (Least active 5-hour period).
        
        Updates feature_df with:
            - L5: Activity level during least active 5-hour period
            - L5_start: Start time of least active period
        """
        if "L5" or "L5_start" not in self.feature_df.columns:
            res = L5(self.enmo)
            self.feature_df["L5"] = res["L5"]
            self.feature_df["L5_start"] = res["L5_start"]

    def get_L5(self):
        """Get computed L5 (Least active 5-hour period) values.
        
        Returns:
            pd.DataFrame: DataFrame containing L5 values representing activity 
                         levels during the least active 5-hour period of each day
        """
        if "L5" not in self.feature_df.columns:
            self.compute_L5()
        return pd.DataFrame(self.feature_df["L5"])

    def compute_M10_start(self):
        """Compute start time of M10 (Most active 10-hour period).
        
        Updates feature_df with:
            - M10: Activity level during most active 10-hour period
            - M10_start: Start time of most active period in minutes from midnight
        """
        if "M10" or "M10_start" not in self.feature_df.columns:
            res = M10(self.enmo)
            self.feature_df["M10"] = res["M10"]
            self.feature_df["M10_start"] = res["M10_start"]

    def get_M10_start(self):
        """Get computed M10 start times.
        
        Returns:
            pd.DataFrame: DataFrame containing M10_start values representing the 
                         start time (in minutes from midnight) of the most active 
                         10-hour period for each day
        """
        if "M10_start" not in self.feature_df.columns:
            self.compute_M10_start()
        return pd.DataFrame(self.feature_df["M10_start"])

    def compute_L5_start(self):
        """Compute start time of L5 (Least active 5-hour period).
        
        Updates feature_df with:
            - L5: Activity level during least active 5-hour period
            - L5_start: Start time of least active period in minutes from midnight
        """
        if "L5" or "L5_start" not in self.feature_df.columns:
            res = L5(self.enmo)
            self.feature_df["L5"] = res["L5"]
            self.feature_df["L5_start"] = res["L5_start"]

    def get_L5_start(self):
        """Get computed L5 start times.
        
        Returns:
            pd.DataFrame: DataFrame containing L5_start values representing the 
                         start time (in minutes from midnight) of the least active 
                         5-hour period for each day
        """
        if "L5_start" not in self.feature_df.columns:
            self.compute_L5_start()
        return pd.DataFrame(self.feature_df["L5_start"])

    def compute_SB(self):
        """Compute Sedentary Behavior (SB) metrics.
        
        Updates feature_df with activity metrics including:
            - SB: Time spent in sedentary behavior
            - LIPA: Time spent in light physical activity
            - MVPA: Time spent in moderate-to-vigorous physical activity
        """
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]

    def get_SB(self):
        """Get computed Sedentary Behavior (SB) values.
        
        Returns:
            pd.DataFrame: DataFrame containing SB values representing time 
                         spent in sedentary behavior for each day
        """
        if "SB" not in self.feature_df.columns:
            self.compute_SB()
        return pd.DataFrame(self.feature_df["SB"])

    def compute_LIPA(self):
        """Compute Light Intensity Physical Activity (LIPA) metrics.
        
        Updates feature_df with activity metrics including:
            - SB: Time spent in sedentary behavior
            - LIPA: Time spent in light physical activity
            - MVPA: Time spent in moderate-to-vigorous physical activity
        """
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]

    def get_LIPA(self):
        """Get computed Light Intensity Physical Activity (LIPA) values.
        
        Returns:
            pd.DataFrame: DataFrame containing LIPA values representing time 
                         spent in light physical activity for each day
        """
        if "LIPA" not in self.feature_df.columns:
            self.compute_LIPA()
        return pd.DataFrame(self.feature_df["LIPA"])

    def compute_MVPA(self):
        """Compute Moderate-to-Vigorous Physical Activity (MVPA) metrics.
        
        Updates feature_df with activity metrics including:
            - SB: Time spent in sedentary behavior
            - LIPA: Time spent in light physical activity
            - MVPA: Time spent in moderate-to-vigorous physical activity
        """
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]

    def get_MVPA(self):
        """Get computed Moderate-to-Vigorous Physical Activity (MVPA) values.
        
        Returns:
            pd.DataFrame: DataFrame containing MVPA values representing time 
                         spent in moderate-to-vigorous physical activity for each day
        """
        if "MVPA" not in self.feature_df.columns:
            self.compute_MVPA()
        return pd.DataFrame(self.feature_df["MVPA"])

    def compute_sleep_predictions(self):
        """Compute binary sleep/wake predictions for each timepoint.
        
        Updates enmo DataFrame with:
            - sleep: Binary values where 1 indicates sleep and 0 indicates wake
        """
        if "sleep" not in self.enmo.columns:
            self.enmo["sleep"] = apply_sleep_wake_predictions(self.enmo, mode="ggir")

    def get_sleep_predictions(self):
        """Get computed sleep/wake predictions.
        
        Returns:
            pd.DataFrame: DataFrame containing binary sleep predictions where 
                         1 indicates sleep and 0 indicates wake
        """
        if "sleep" not in self.enmo.columns:
            self.compute_sleep_predictions()
        return pd.DataFrame(self.enmo["sleep"])

    def compute_TST(self):
        """Compute Total Sleep Time (TST).
        
        First ensures sleep predictions are computed, then calculates total
        sleep time for each day.
        """
        if "sleep" not in self.enmo.columns:
            self.enmo["sleep"] = apply_sleep_wake_predictions(self.enmo)
        if "TST" not in self.feature_df.columns:
            self.feature_df["TST"] = tst(self.enmo)    

    def get_TST(self):
        """Get computed Total Sleep Time (TST) values.
        
        Returns:
            pd.DataFrame: DataFrame containing TST values representing total
                         sleep time for each day
        """
        if "TST" not in self.feature_df.columns:
            self.compute_TST()
        return pd.DataFrame(self.feature_df["TST"])

    def compute_WASO(self):
        """Compute Wake After Sleep Onset (WASO).
        
        First ensures sleep predictions are computed, then calculates wake
        time occurring after initial sleep onset.
        """
        if "sleep" not in self.enmo.columns:
            self.enmo["sleep"] = apply_sleep_wake_predictions(self.enmo)
        if "WASO" not in self.feature_df.columns:
            self.feature_df["WASO"] = waso(self.enmo)

    def get_WASO(self):
        """Get computed Wake After Sleep Onset (WASO) values.
        
        Returns:
            pd.DataFrame: DataFrame containing WASO values representing wake
                         time after sleep onset for each day
        """
        if "WASO" not in self.feature_df.columns:
            self.compute_WASO()
        return pd.DataFrame(self.feature_df["WASO"])

    def compute_PTA(self):
        """Compute Prolonged Time Awake (PTA).
        
        First ensures sleep predictions are computed, then calculates
        periods of prolonged wakefulness.
        """
        if "sleep" not in self.enmo.columns:
            self.enmo["sleep"] = apply_sleep_wake_predictions(self.enmo)
        if "PTA" not in self.feature_df.columns:
            self.feature_df["PTA"] = pta(self.enmo)

    def get_PTA(self):
        """Get computed Prolonged Time Awake (PTA) values.
        
        Returns:
            pd.DataFrame: DataFrame containing PTA values representing periods
                         of prolonged wakefulness for each day
        """
        if "PTA" not in self.feature_df.columns:
            self.compute_PTA()
        return pd.DataFrame(self.feature_df["PTA"])

    def compute_SRI(self):
        """Compute Sleep Regularity Index (SRI).
        
        First ensures sleep predictions are computed, then calculates
        the consistency of sleep timing between days.
        """
        if "sleep" not in self.enmo.columns:
            self.enmo["sleep"] = apply_sleep_wake_predictions(self.enmo)
            print("computed sleep predictions")
        if "SRI" not in self.feature_df.columns:
            self.feature_df["SRI"] = sri(self.enmo)

    def get_SRI(self):
        """Get computed Sleep Regularity Index (SRI) values.
        
        Returns:
            pd.DataFrame: DataFrame containing SRI values representing the
                         consistency of sleep timing between days
        """
        if "SRI" not in self.feature_df.columns:
            self.compute_SRI()
        return pd.DataFrame(self.feature_df["SRI"])

    def get_all(self):
        """Returns the entire feature DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all computed features
        """
        return self.feature_df, self.feature_dict

    def get_enmo_data(self):
        """Returns the raw ENMO data.
        
        Returns:
            pd.DataFrame: DataFrame containing ENMO data with datetime index
        """
        return self.enmo
