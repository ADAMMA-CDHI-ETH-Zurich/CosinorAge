import pandas as pd
import matplotlib.pyplot as plt

from ..dataloaders import DataLoader
from .utils.nonparam_analysis import *
from .utils.physical_activity_metrics import *
from .utils.sleep_metrics import *
from .utils.cosinor_analysis import *


# TODO: Implement the WearableFeatures class
class WearableFeatures:

    def __init__(self, loader: DataLoader):

        self.enmo = loader.get_enmo_data().copy()

        self.feature_df = pd.DataFrame(index=pd.unique(self.enmo.index.date))
        self.feature_dict = {}

    def get_cosinor_features(self):
        if "cosinor_features" not in self.feature_dict.keys():
            params, fitted = cosinor(self.enmo)
            self.feature_df["MESOR"] = params["MESOR"]
            self.feature_df["amplitude"] = params["amplitude"] 
            self.feature_df["acrophase"] = params["acrophase"]
            self.feature_df["acrophase_time"] = params["acrophase_time"]
            self.enmo["cosinor_fitted"] = fitted
        return pd.DataFrame(self.feature_df[["MESOR", "amplitude", "acrophase", "acrophase_time"]])
        
    def get_IV(self):
        if "IV" not in self.feature_df.columns:
            self.feature_df["IV"] = IV(self.enmo)
        return pd.DataFrame(self.feature_df["IV"])

    def get_IS(self):
        if "IS" not in self.feature_df.columns:
            self.feature_df["IS"] = IS(self.enmo)
        return pd.DataFrame(self.feature_df["IS"])

    def get_RA(self):
        if "RA" not in self.feature_df.columns:
            self.feature_df["RA"] = RA(self.enmo)
        return pd.DataFrame(self.feature_df["RA"])

    def get_M10(self):
        if "M10" or "M10_start" not in self.feature_df.columns:
            res = M10(self.enmo)
            self.feature_df["M10"] = res["M10"]
            self.feature_df["M10_start"] = res["M10_start"]
        return pd.DataFrame(self.feature_df["M10"])

    def get_L5(self):
        if "L5" or "L5_start" not in self.feature_df.columns:
            res = L5(self.enmo)
            self.feature_df["L5"] = res["L5"]
            self.feature_df["L5_start"] = res["L5_start"]
        return pd.DataFrame(self.feature_df["L5"])

    def get_M10_start(self):
        if "M10" or "M10_start" not in self.feature_df.columns:
            res = M10(self.enmo)
            self.feature_df["M10"] = res["M10"]
            self.feature_df["M10_start"] = res["M10_start"]
        return pd.DataFrame(self.feature_df["M10_start"])

    def get_L5_start(self):
        if "L5" or "L5_start" not in self.feature_df.columns:
            res = L5(self.enmo)
            self.feature_df["L5"] = res["L5"]
            self.feature_df["L5_start"] = res["L5_start"]
        return pd.DataFrame(self.feature_df["L5_start"])

    def get_SB(self):
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]
        return pd.DataFrame(self.feature_df["SB"])

    def get_LIPA(self):
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]
        return pd.DataFrame(self.feature_df["LIPA"])

    def get_MVPA(self):
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]
        return pd.DataFrame(self.feature_df["MVPA"])

    def get_sleep_predictions(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        return pd.DataFrame(self.enmo["sleep_predictions"])

    def get_TST(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "TST" not in self.feature_df.columns:
            self.feature_df["TST"] = tst(self.enmo)    
        return pd.DataFrame(self.feature_df["TST"])

    def get_WASO(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "WASO" not in self.feature_df.columns:
            self.feature_df["WASO"] = waso(self.enmo)
        return pd.DataFrame(self.feature_df["WASO"])

    def get_PTA(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "PTA" not in self.feature_df.columns:
            self.feature_df["PTA"] = pta(self.enmo)
        return pd.DataFrame(self.feature_df["PTA"])

    def get_SRI(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "SRI" not in self.feature_df.columns:
            self.feature_df["SRI"] = sri(self.enmo)
        return pd.DataFrame(self.feature_df["SRI"])

    def get_all(self):
        """Returns the entire feature DataFrame."""
        return self.feature_df

    def get_enmo_data(self):
        return self.enmo

    def plot_sleep_predictions(self):
        plt.figure(figsize=(20, 0.5))
        plt.plot(self.enmo["sleep_predictions"] == 0, 'r.')
        plt.plot(self.enmo["sleep_predictions"] == 1, 'g.')
        plt.ylim(1, 1)
        plt.yticks([])
        plt.show()

    def plot_cosinor(self):
        minutes = np.arange(1, 1441)

        # for each day, plot the ENMO and the cosinor fit
        for date, group in self.enmo.groupby(self.enmo.index.date):
            plt.figure(figsize=(20, 2))
            plt.plot(minutes, group["ENMO"], 'r-')
            # cosinor fit based on the parameters from cosinor()
            plt.plot(minutes, group["cosinor_fitted"], 'b-')
            plt.show()

