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


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colormaps
import numpy as np
import pandas as pd
from datetime import timedelta

def dashboard(features):
    """
    Generate a comprehensive visualization dashboard for accelerometer data analysis.
    
    This function creates multiple plots to visualize various aspects of the accelerometer data:
    1. Cosinor fit plot showing ENMO data with mesor, amplitude, and acrophase
    2. Daily ENMO plots with M10 and L5 periods highlighted
    3. IS (Inter-daily Stability) and IV (Intra-daily Variability) visualization
    4. RA (Relative Amplitude) plots for each day
    5. Sleep predictions visualization
    6. Sleep metrics (TST, WASO, PTA, NWB, SOL) visualization
    7. Physical activity breakdown by intensity levels

    Parameters
    ----------
    features : Features
        A Features object containing all extracted features and raw data.
        Expected to have the following attributes:
        - feature_dict: Dictionary containing cosinor, nonparam, sleep, and physical_activity features
        - get_ml_data(): Method returning DataFrame with ENMO and cosinor_fitted data
        - get_features(): Method returning all extracted features

    Returns
    -------
    None
        Displays multiple matplotlib figures using plt.show()
    """
    
    data = features.get_ml_data()

    mesor = features.feature_dict['cosinor']['mesor']
    amplitude = features.feature_dict['cosinor']['amplitude']
    acrophase = features.feature_dict['cosinor']['acrophase']
    acrophase_time = features.feature_dict['cosinor']['acrophase_time']

    timestamps = data.index
    plt.figure(figsize=(22.5, 5))
    plt.plot(timestamps, data["ENMO"], '-', color='grey', alpha=0.8, label="ENMO")
    plt.plot(timestamps, data["cosinor_fitted"], 'b-', label="Cosinor Fit")
    plt.ylim(0, max(data["ENMO"])*1.05)
    plt.axhline(mesor, color='green', linestyle='--', label='MESOR')
    plt.axvline(timestamps.min() + timedelta(minutes=acrophase_time), color='purple', linestyle=':', label='Acrophase Time')
    plt.text(timestamps.min() + timedelta(minutes=acrophase_time), 0.95*max(data["ENMO"]), f'{acrophase_time:.1f} minutes ({acrophase:.1f}°)', color='purple', horizontalalignment='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
    plt.text(timestamps.min()-timedelta(minutes=120), mesor, f'{mesor:.1f} mg', color='green', horizontalalignment='right', verticalalignment='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
    plt.vlines(timestamps.max()+timedelta(minutes=20), mesor, mesor+amplitude, color='red', linestyle='-', label='Amplitude')
    plt.text(timestamps.max()+timedelta(minutes=120), mesor+(amplitude/2), f'{amplitude:.1f} mg', color='red', horizontalalignment='left', verticalalignment='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))


    plt.title(f"Cosinor Fit")
    plt.xlabel("Timestamp")
    plt.ylabel("ENMO (mg)")
    plt.legend(ncol=5)
    plt.tight_layout()

    # Extract data and group by date
    days = data.groupby(data.index.date)
    
    # Extract features
    feature_data = features.get_features()
    m10_start = feature_data['nonparam']['M10_start']
    l5_start = feature_data['nonparam']['L5_start']
    m10_values = feature_data['nonparam']['M10']  # Assuming these values exist
    l5_values = feature_data['nonparam']['L5']  # Assuming these values exist

    num_days = len(days)
    fig, axes = plt.subplots(num_days, 2, figsize=(18, 1.5 * num_days), gridspec_kw={"width_ratios": [4, 1]})

    # Define fixed y-range for ENMO plots
    enmo_y_range = (-20, data['ENMO'].max())

    for i, (day, day_data) in enumerate(days):
        # Lineplot of ENMO
        sns.lineplot(data=day_data, x=day_data.index, y="ENMO", ax=axes[i, 0], linewidth=0.8, color="grey", alpha=0.8)
        
        # Highlight M10 and L5 periods
        axes[i, 0].axvspan(m10_start[i], m10_start[i] + pd.Timedelta(hours=10), color="red", alpha=0.3, label="M10 Period")
        axes[i, 0].axvspan(l5_start[i], l5_start[i] + pd.Timedelta(hours=5), color="green", alpha=0.3, label="L5 Period")
        
        axes[i, 0].set_title(f"ENMO Time Series - {day}")
        axes[i, 0].set_ylabel("ENMO (mg)")
        axes[i, 0].set_xlabel("Timestamp")
        axes[i, 0].set_ylim(enmo_y_range)
        
        axes[i, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i, 0].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        # Add legend only once
        if i == 0:
            axes[i, 0].legend(loc="upper right")

        # Bar plot for M10 and L5 values
        bar_plot = sns.barplot(
            x=["M10", "L5"], 
            y=[m10_values[i], l5_values[i]], 
            hue=["M10", "L5"], 
            ax=axes[i, 1], 
            palette=["red", "green"],
            dodge=False,
            legend=False
        )
        axes[i, 1].set_title("M10 & L5 Values")
        axes[i, 1].set_ylabel("Mean ENMO (mg)")
        axes[i, 1].set_ylim(0, max(max(m10_values), max(l5_values))*1.4)

        # Annotate bar plot with values
        colors = ['red', 'green']
        for p, color in zip(bar_plot.patches, colors):
            bar_plot.annotate(f'{p.get_height():.2f}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='center', 
                              fontsize=10, color=color, 
                              xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()


    fig, axes = plt.subplots(1, 2, figsize=(22.5, 0.4))
    # IS plot
    axes[0].hlines(y=0, xmin=0, xmax=feature_data['nonparam']['IS'], color="blue", alpha=0.8, linewidth=2)
    axes[0].hlines(y=0, xmin=feature_data['nonparam']['IS'], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
    axes[0].scatter(feature_data['nonparam']['IS'], 0, color="blue", s=100, marker="o", label="IS Value")
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_title("IS Value")

    # IV plot
    axes[1].hlines(y=0, xmin=0, xmax=feature_data['nonparam']['IV'], color="green", alpha=0.8, linewidth=2)
    if feature_data['nonparam']['IV'] < 2:
        axes[1].hlines(y=0, xmin=feature_data['nonparam']['IV'], xmax=2, color="grey", alpha=0.8, linewidth=2, zorder=1)
    else:
        axes[1].hlines(y=0, xmin=feature_data['nonparam']['IV'], xmax=3, color="grey", alpha=0.8, linewidth=2, zorder=1)

    if feature_data['nonparam']['IV'] < 3:
        axes[1].hlines(y=0, xmin=2, xmax=3, color="grey", alpha=0.8, linewidth=2, zorder=1)

    axes[1].vlines(x=2, ymin=-0.01, ymax=0.01, color="red", alpha=0.8, linewidth=2, zorder=1)
    axes[1].scatter(feature_data['nonparam']['IV'], 0, color="green", s=100, marker="o", label="IV Value")
    if feature_data['nonparam']['IV'] < 3:
        axes[1].set_xlim(0, 3)
    else:
        axes[1].set_xlim(0, feature_data['nonparam']['IV'])
    axes[1].set_yticks([])
    axes[1].set_title("Intradaily Variability (IV)")
    #plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(18, 0.6*num_days), sharex=True)

    # RA subplot
    dates = pd.unique(data.index.date)
    positions = np.arange(len(dates))[::-1]
    for i, date in enumerate(dates):
        ax.hlines(y=positions[i], xmin=0, xmax=feature_data['nonparam']['RA'][i], color="orange", alpha=0.8, linewidth=2)
        ax.hlines(y=positions[i], xmin=feature_data['nonparam']['RA'][i], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
        ax.scatter(feature_data['nonparam']['RA'][i], positions[i], color="orange", s=100, marker="o", label="RA Value" if i == 0 else "")

    # Setting y-ticks as the dates
    ax.set_yticks(positions, dates)
    ax.set_xlim(0, 1)  # Adjust this limit based on your data
    ax.set_title("Relative Amplitude (RA)")
    ax.set_xlabel("RA")
    ax.set_ylabel("Day")
    plt.tight_layout()

    # Sleep predictions
    fig, axes = plt.subplots(num_days, figsize=(18, 1.5 * num_days))
    for i, (day, day_data) in enumerate(days):
        # Lineplot of ENMO
        sns.lineplot(data=day_data, x=day_data.index, y="ENMO", ax=axes[i], linewidth=0.8, color="grey", alpha=0.8)
        
        # highlight sleep predictions
        axes[i].fill_between(day_data.index, day_data['sleep']*enmo_y_range[1], color='blue', alpha=0.5, label='Sleep')

        # highlight non-wear periods
        if 'wear' in day_data.columns and day_data['wear'].max() != -1:
            axes[i].fill_between(day_data.index, (1-day_data['wear'])*enmo_y_range[1], color='red', alpha=0.5, label='Non-wear')
        
        axes[i].set_title(f"Sleep Predictions - {day}")
        axes[i].set_ylabel("ENMO (mg)")
        axes[i].set_xlabel("Timestamp")
        axes[i].set_ylim(enmo_y_range)
        
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        # Add legend only once
        if i == 0:
            axes[i].legend(loc="upper right")

    plt.tight_layout()

    # Sleep metrics subplot
    tst = feature_data['sleep']['TST']
    waso = feature_data['sleep']['WASO']
    pta = feature_data['sleep']['PTA']
    nwb = feature_data['sleep']['NWB']
    sol = feature_data['sleep']['SOL']

    positions = np.arange(len(dates))[::-1]

    fig, ax = plt.subplots(1, 5, figsize=(18, 0.6*num_days))
    for i, date in enumerate(dates):
        ax[0].hlines(y=positions[i], xmin=0, xmax=tst[i], color="orange", alpha=0.8, linewidth=2)
        ax[0].hlines(y=positions[i], xmin=tst[i], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
        ax[0].scatter(tst[i], positions[i], color="orange", s=100, marker="o", label="TST" if i == 0 else "")
    ax[0].set_yticks(positions, dates)
    ax[0].set_title("Total Sleep Time (TST)")
    ax[0].set_ylabel("Day")
    ax[0].set_xlabel("TST (in minutes)")

    for i, date in enumerate(dates):
        ax[1].hlines(y=positions[i], xmin=0, xmax=waso[i], color="orange", alpha=0.8, linewidth=2)
        ax[1].hlines(y=positions[i], xmin=waso[i], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
        ax[1].scatter(waso[i], positions[i], color="orange", s=100, marker="o", label="WASO" if i == 0 else "")
    ax[1].set_yticks(positions, dates)
    ax[1].set_title("Wake After Sleep Onset (WASO)")
    ax[1].set_xlabel("WASO (in minutes)")
    
    for i, date in enumerate(dates):
        ax[2].hlines(y=positions[i], xmin=0, xmax=pta[i], color="orange", alpha=0.8, linewidth=2)
        ax[2].hlines(y=positions[i], xmin=pta[i], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
        ax[2].scatter(pta[i], positions[i], color="orange", s=100, marker="o", label="PTA" if i == 0 else "")
    ax[2].set_yticks(positions, dates)
    ax[2].set_title("Percent Time Asleep (PTA)")
    ax[2].set_xlabel("PTA (in %)")
    for i, date in enumerate(dates):
        ax[3].hlines(y=positions[i], xmin=0, xmax=nwb[i], color="orange", alpha=0.8, linewidth=2)
        ax[3].hlines(y=positions[i], xmin=nwb[i], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
        ax[3].scatter(nwb[i], positions[i], color="orange", s=100, marker="o", label="NWB" if i == 0 else "")
    ax[3].set_yticks(positions, dates)
    ax[3].set_title("Number of Waking Bouts (NWB)")
    ax[3].set_xlabel("NWB")

    for i, date in enumerate(dates):
        ax[4].hlines(y=positions[i], xmin=0, xmax=sol[i], color="orange", alpha=0.8, linewidth=2)
        ax[4].hlines(y=positions[i], xmin=sol[i], xmax=1, color="grey", alpha=0.8, linewidth=2, zorder=1)
        ax[4].scatter(sol[i], positions[i], color="orange", s=100, marker="o", label="SOL" if i == 0 else "")
    ax[4].set_yticks(positions, dates)
    ax[4].set_title("Sleep Onset Latency (SOL)")
    ax[4].set_xlabel("SOL (in minutes)")
    plt.tight_layout()

    sedentary = feature_data['physical_activity']['sedentary']
    light = feature_data['physical_activity']['light']
    moderate = feature_data['physical_activity']['moderate']
    vigorous = feature_data['physical_activity']['vigorous']

    # Creating a horizontal stacked bar chart
    viridis = colormaps.get_cmap('viridis')  # Correct usage without specifying the number of discrete colors
    colors = [viridis(0.2), viridis(0.4), viridis(0.6), viridis(0.8)]  # Manually picking shades
    positions = np.arange(len(dates))[::-1]

    plt.figure(figsize=(18, 3.5))
    plt.barh(positions, sedentary, height=0.8, label='Sedentary', color=colors[0])
    plt.barh(positions, light, height=0.8, left=sedentary, label='Light', color=colors[1])
    plt.barh(positions, moderate, height=0.8, left=np.array(sedentary) + np.array(light), label='Moderate', color=colors[2])
    plt.barh(positions, vigorous, height=0.8, left=np.array(sedentary) + np.array(light) + np.array(moderate), label='Vigorous', color=colors[3])

    # Adding labels and title
    plt.ylabel('Days')
    plt.xlabel('Minutes')
    plt.title('Daily Activity Breakdown')
    plt.yticks(positions, dates)
    plt.legend(loc='upper right', bbox_to_anchor=(1, -0.5), ncol=4, title="Activity Levels", frameon=True)

    plt.tight_layout()

    plt.show()



