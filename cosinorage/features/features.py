import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

    def run(self):
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
        cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]
        if not all(col in self.feature_df.columns for col in cosinor_columns) or "cosinor_fitted" not in self.enmo.columns:
            params, fitted = cosinor(self.enmo)
            self.feature_df["MESOR"] = params["MESOR"]
            self.feature_df["amplitude"] = params["amplitude"]
            self.feature_df["acrophase"] = params["acrophase"]
            self.feature_df["acrophase_time"] = params["acrophase_time"]
            self.enmo["cosinor_fitted"] = fitted

    def get_cosinor_features(self):
        cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]
        if not all(col in self.feature_df.columns for col in cosinor_columns):
            self.compute_cosinor_features()
        return pd.DataFrame(self.feature_df[cosinor_columns])
        
    def compute_IV(self):
        if "IV" not in self.feature_df.columns:
            self.feature_df["IV"] = IV(self.enmo)

    def get_IV(self):
        if "IV" not in self.feature_df.columns:
            self.compute_IV()
        return pd.DataFrame(self.feature_df["IV"])

    def compute_IS(self):
        if "IS" not in self.feature_df.columns:
            self.feature_df["IS"] = IS(self.enmo)

    def get_IS(self):
        if "IS" not in self.feature_df.columns:
            self.compute_IS()
        return pd.DataFrame(self.feature_df["IS"])

    def compute_RA(self):
        if "RA" not in self.feature_df.columns:
            self.feature_df["RA"] = RA(self.enmo)

    def get_RA(self):
        if "RA" not in self.feature_df.columns:
            self.compute_RA()
        return pd.DataFrame(self.feature_df["RA"])

    def compute_M10(self):
        if "M10" or "M10_start" not in self.feature_df.columns:
            res = M10(self.enmo)
            self.feature_df["M10"] = res["M10"]
            self.feature_df["M10_start"] = res["M10_start"]

    def get_M10(self):
        if "M10" not in self.feature_df.columns:
            self.compute_M10()
        return pd.DataFrame(self.feature_df["M10"])

    def compute_L5(self):
        if "L5" or "L5_start" not in self.feature_df.columns:
            res = L5(self.enmo)
            self.feature_df["L5"] = res["L5"]
            self.feature_df["L5_start"] = res["L5_start"]

    def get_L5(self):
        if "L5" not in self.feature_df.columns:
            self.compute_L5()
        return pd.DataFrame(self.feature_df["L5"])

    def compute_M10_start(self):
        if "M10" or "M10_start" not in self.feature_df.columns:
            res = M10(self.enmo)
            self.feature_df["M10"] = res["M10"]
            self.feature_df["M10_start"] = res["M10_start"]

    def get_M10_start(self):
        if "M10_start" not in self.feature_df.columns:
            self.compute_M10_start()
        return pd.DataFrame(self.feature_df["M10_start"])

    def compute_L5_start(self):
        if "L5" or "L5_start" not in self.feature_df.columns:
            res = L5(self.enmo)
            self.feature_df["L5"] = res["L5"]
            self.feature_df["L5_start"] = res["L5_start"]

    def get_L5_start(self):
        if "L5_start" not in self.feature_df.columns:
            self.compute_L5_start()
        return pd.DataFrame(self.feature_df["L5_start"])

    def compute_SB(self):
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]

    def get_SB(self):
        if "SB" not in self.feature_df.columns:
            self.compute_SB()
        return pd.DataFrame(self.feature_df["SB"])

    def compute_LIPA(self):
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]

    def get_LIPA(self):
        if "LIPA" not in self.feature_df.columns:
            self.compute_LIPA()
        return pd.DataFrame(self.feature_df["LIPA"])

    def compute_MVPA(self):
        if "SB" or "LIPA" or "MVPA" not in self.feature_df.columns:
            res = activity_metrics(self.enmo)
            self.feature_df["SB"] = res["SB"]
            self.feature_df["LIPA"] = res["LIPA"]
            self.feature_df["MVPA"] = res["MVPA"]

    def get_MVPA(self):
        if "MVPA" not in self.feature_df.columns:
            self.compute_MVPA()
        return pd.DataFrame(self.feature_df["MVPA"])

    def compute_sleep_predictions(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)

    def get_sleep_predictions(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.compute_sleep_predictions()
        return pd.DataFrame(self.enmo["sleep_predictions"])

    def compute_TST(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "TST" not in self.feature_df.columns:
            self.feature_df["TST"] = tst(self.enmo)    

    def get_TST(self):
        if "TST" not in self.feature_df.columns:
            self.compute_TST()
        return pd.DataFrame(self.feature_df["TST"])

    def compute_WASO(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "WASO" not in self.feature_df.columns:
            self.feature_df["WASO"] = waso(self.enmo)

    def get_WASO(self):
        if "WASO" not in self.feature_df.columns:
            self.compute_WASO()
        return pd.DataFrame(self.feature_df["WASO"])

    def compute_PTA(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "PTA" not in self.feature_df.columns:
            self.feature_df["PTA"] = pta(self.enmo)

    def get_PTA(self):
        if "PTA" not in self.feature_df.columns:
            self.compute_PTA()
        return pd.DataFrame(self.feature_df["PTA"])

    def compute_SRI(self):
        if "sleep_predictions" not in self.enmo.columns:
            self.enmo["sleep_predictions"] = apply_sleep_wake_predictions(self.enmo)
        if "SRI" not in self.feature_df.columns:
            self.feature_df["SRI"] = sri(self.enmo)

    def get_SRI(self):
        if "SRI" not in self.feature_df.columns:
            self.compute_SRI()
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
        if "cosinor_fitted" not in self.enmo.columns:
            raise ValueError("Cosinor fitted values not computed.")

        minutes = np.arange(0, 1440)
        timestamps = pd.date_range("00:00", "23:59", freq="1min")

        # for each day, plot the ENMO and the cosinor fit
        for date, group in self.enmo.groupby(self.enmo.index.date):
            plt.figure(figsize=(20, 10))
            plt.plot(minutes, group["ENMO"]*1000, 'r-')
            # cosinor fit based on the parameters from cosinor()
            plt.plot(minutes, group["cosinor_fitted"]*1000, 'b-')
            plt.ylim(0, max(group["ENMO"]*1000)*1.5)
            plt.xlim(0, 1600)

            plt.title(date)

            plt.xticks(minutes[::60])  # Tick every hour
            plt.gca().xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: timestamps[int(x)].strftime("%H:%M") if 0 <= int(x) < 1440 else "")
        )

            # x ticks should be daytime hours
            plt.axhline(self.feature_df.loc[date, "MESOR"]*1000, color='green', linestyle='--', label='Mesor')
            plt.text(minutes[0]-80, self.feature_df.loc[date, "MESOR"]*1000, f'Mesor: {(self.feature_df.loc[date, "MESOR"]*1000):.2f}', color='green', fontsize=8, va='center')

            cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]
            if all(col in self.feature_df.columns for col in cosinor_columns):
                plt.hlines(
                    y=max(group["ENMO"]*1000)*1.25, 
                    xmin=0, 
                    xmax=self.feature_df.loc[date, "acrophase_time"], 
                    color='black', linewidth=1, label='Acrophase Time'
                )

                plt.vlines(
                    [0, self.feature_df.loc[date, "acrophase_time"]], 
                    ymin=max(group["ENMO"]*1000)*1.25-2, 
                    ymax=max(group["ENMO"]*1000)*1.25+2, 
                    color='black', linewidth=1
                )
                plt.text(
                    self.feature_df.loc[date, "acrophase_time"]/2, 
                    max(group["ENMO"]*1000)*1.25+5, 
                    f'Acrophase Time: {self.feature_df.loc[date, "acrophase_time"]/60:.2f} h', 
                    color='black', fontsize=8, ha='center'
                )

                plt.vlines(
                    x=1445, 
                    ymin=self.feature_df.loc[date, "MESOR"]*1000, 
                    ymax=self.feature_df.loc[date, "MESOR"]*1000+self.feature_df.loc[date, "amplitude"]*1000, 
                    color='black', linewidth=1, label='Amplitude'
                )
                plt.hlines(
                    y=[self.feature_df.loc[date, "MESOR"]*1000, self.feature_df.loc[date, "MESOR"]*1000+self.feature_df.loc[date, "amplitude"]*1000], 
                    xmin=1445 - 4, 
                    xmax=1445 + 4, 
                        color='black', linewidth=1
                    )
                plt.text(
                    1450, 
                self.feature_df.loc[date, "MESOR"]*1000+self.feature_df.loc[date, "amplitude"]/2*1000, 
                f'Amplitude: {self.feature_df.loc[date, "amplitude"] * 1000:.2f}', 
                        color='black', fontsize=8, va='center'
                    )

            plt.show()

