import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_sleep_predictions(feature_obj, simple=True, start_date=None, end_date=None):
    """Plot sleep predictions over time.
    Args:
        simple (bool, optional): If True, shows simple binary plot. If False, shows detailed plot. Defaults to True.
        start_date (datetime, optional): Start date for plotting. Defaults to None (earliest date).
        end_date (datetime, optional): End date for plotting. Defaults to None (latest date).
    """
    if start_date is None:
        start_date = feature_obj.ml_data.index[0]
    if end_date is None:
        end_date = feature_obj.ml_data.index[-1]
    selected_data = feature_obj.ml_data[(feature_obj.ml_data.index >= start_date) & (feature_obj.ml_data.index <= end_date)]
    if simple:
        plt.figure(figsize=(20, 0.5))
        plt.plot(selected_data["sleep"] == 0, 'g.', label='Wake')
        plt.plot(selected_data["sleep"] == 1, 'b.', label='Sleep')
        if 'wear' in selected_data.columns:
            plt.plot(selected_data["wear"] != 1, 'r.', label='Non-wear')
        plt.ylim(0.9, 1.1)
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(30, 6))
        # plot sleep predictions as red bands
        plt.fill_between(selected_data.index, (1-selected_data['sleep'])*1000, color='green', alpha=0.5, label='Wake')
        plt.fill_between(selected_data.index, selected_data['sleep']*1000, color='blue', alpha=0.5, label='Sleep')
        if 'wear' in selected_data.columns:
            plt.fill_between(selected_data.index, (1-selected_data['wear'])*1000, color='red', alpha=0.5, label='Non-wear')
        plt.plot(selected_data['ENMO'], label='ENMO', color='black')
        # y axis limits
        plt.ylim(0, max(selected_data['ENMO'])*1.25)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("ENMO")
        plt.show()

def plot_non_wear(feature_obj, simple=True, start_date=None, end_date=None):
    """Plot sleep predictions over time.
    Args:
        simple (bool, optional): If True, shows simple binary plot. If False, shows detailed plot. Defaults to True.
        start_date (datetime, optional): Start date for plotting. Defaults to None (earliest date).
        end_date (datetime, optional): End date for plotting. Defaults to None (latest date).
    """
    if start_date is None:
        start_date = feature_obj.ml_data.index[0]
    if end_date is None:
        end_date = feature_obj.ml_data.index[-1]
    selected_data = feature_obj.ml_data[(feature_obj.ml_data.index >= start_date) & (feature_obj.ml_data.index <= end_date)]
    if simple:
        plt.figure(figsize=(20, 0.5))
        plt.plot(selected_data["wear"] == 1, 'g.', label='Wear')
        plt.plot(selected_data["wear"] == 0, 'r.', label='Non-wear')
        plt.ylim(0.9, 1.1)
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(30, 6))
        plt.plot(selected_data['ENMO'], label='ENMO', color='black')
        # plot sleep predictions as red bands
        plt.fill_between(selected_data.index, (1-selected_data['wear'])*1000, color='red', alpha=0.5, label='Non-wear')
        plt.fill_between(selected_data.index, selected_data['wear']*1000, color='green', alpha=0.5, label='Wear')
        # y axis limits
        plt.ylim(0, max(selected_data['ENMO'])*1.25)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("ENMO")
        plt.show()

def plot_cosinor(feature_obj, multiday=True):
    """Plot cosinor analysis results for each day.
    
    Creates plots showing:
        - Raw ENMO data
        - Fitted cosinor curve
        - MESOR line
        - Amplitude visualization
        - Acrophase time marker
        
    Raises:
        ValueError: If cosinor features haven't been computed
    """
    if multiday:
        if "cosinor_multiday_fitted" not in feature_obj.ml_data.columns:
            raise ValueError("Multiday cosinor fitted values not computed.")
        minutes = np.arange(0, len(feature_obj.ml_data))
        timestamps = feature_obj.ml_data.index
        plt.figure(figsize=(20, 10))
        plt.plot(timestamps, feature_obj.ml_data["ENMO"], 'r-')
        plt.plot(timestamps, feature_obj.ml_data["cosinor_multiday_fitted"], 'b-')
        plt.ylim(0, max(feature_obj.ml_data["ENMO"])*1.5)
        cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]
        if all(col in feature_obj.feature_df.columns for col in cosinor_columns):
            # x ticks should be daytime hours
            plt.axhline(feature_obj.feature_dict["MESOR"], color='green', linestyle='--', label='MESOR')
    else:
        if "cosinor_by_day_fitted" not in feature_obj.ml_data.columns:
            raise ValueError("By-day cosinor fitted values not computed.")
        minutes = np.arange(0, 1440)
        timestamps = pd.date_range("00:00", "23:59", freq="1min")
        # for each day, plot the ENMO and the cosinor fit
        for date, group in feature_obj.ml_data.groupby(feature_obj.ml_data.index.date):
            plt.figure(figsize=(20, 10))
            plt.plot(minutes, group["ENMO"], 'r-')
            # cosinor fit based on the parameters from cosinor()
            plt.plot(minutes, group["cosinor_by_day_fitted"], 'b-')
            plt.ylim(0, max(group["ENMO"])*1.5)
            plt.xlim(0, 1600)
            plt.title(date)
            plt.xticks(minutes[::60])  # Tick every hour
            plt.gca().xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: timestamps[int(x)].strftime("%H:%M") if 0 <= int(x) < 1440 else "")
            )
            cosinor_columns = ["MESOR", "amplitude", "acrophase", "acrophase_time"]
            if all(col in feature_obj.feature_df.columns for col in cosinor_columns):
                # x ticks should be daytime hours
                plt.axhline(feature_obj.feature_df.loc[date, "MESOR"], color='green', linestyle='--', label='MESOR')
                plt.text(minutes[0]-105, feature_obj.feature_df.loc[date, "MESOR"], f'MESOR: {(feature_obj.feature_df.loc[date, "MESOR"]):.2f}mg', color='green', fontsize=8, va='center')
                plt.hlines(
                    y=max(group["ENMO"])*1.25, 
                    xmin=0, 
                    xmax=feature_obj.feature_df.loc[date, "acrophase_time"], 
                    color='black', linewidth=1, label='Acrophase Time'
                )
                plt.vlines(
                    [0, feature_obj.feature_df.loc[date, "acrophase_time"]], 
                    ymin=max(group["ENMO"])*1.25-2, 
                    ymax=max(group["ENMO"])*1.25+2, 
                    color='black', linewidth=1
                )
                plt.text(
                    feature_obj.feature_df.loc[date, "acrophase_time"]/2, 
                    max(group["ENMO"])*1.25+2, 
                    f'Acrophase Time: {feature_obj.feature_df.loc[date, "acrophase_time"]/60:.2f}h', 
                    color='black', fontsize=8, ha='center'
                )
                plt.vlines(
                    x=1445, 
                    ymin=feature_obj.feature_df.loc[date, "MESOR"], 
                    ymax=feature_obj.feature_df.loc[date, "MESOR"]+feature_obj.feature_df.loc[date, "amplitude"], 
                    color='black', linewidth=1, label='Amplitude'
                )
                plt.hlines(
                    y=[feature_obj.feature_df.loc[date, "MESOR"], feature_obj.feature_df.loc[date, "MESOR"]+feature_obj.feature_df.loc[date, "amplitude"]], 
                    xmin=1445 - 4, 
                    xmax=1445 + 4, 
                    color='black', linewidth=1
                )
                plt.text(
                    1450, 
                    feature_obj.feature_df.loc[date, "MESOR"]+feature_obj.feature_df.loc[date, "amplitude"]/2, 
                    f'Amplitude: {feature_obj.feature_df.loc[date, "amplitude"] :.2f}mg', 
                    color='black', fontsize=8, va='center'
                )
    plt.show()