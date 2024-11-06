import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .loader import DataLoader


def plot_enmo(loader: DataLoader):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=loader.get_enmo_per_minute(), x='TIMESTAMP', y='ENMO')
    plt.xlabel('Time')
    plt.ylabel('ENMO')
    plt.title('ENMO per Minute')
    plt.xticks(rotation=45)
    plt.show()

def plot_enmo_difference(loader_1: DataLoader, loader_2: DataLoader):

    df1 = loader_1.get_enmo_per_minute()
    df2 = loader_2.get_enmo_per_minute()

    #dertermine overlapping time period
    starttime = max(df1['TIMESTAMP'].min(), df2['TIMESTAMP'].min())
    endtime = min(df1['TIMESTAMP'].max(), df2['TIMESTAMP'].max())

    # handle if no overlap
    if starttime > endtime:
        raise ValueError("No overlapping time period found.")

    # filter data to overlapping time period
    df1 = df1[(df1['TIMESTAMP'] >= starttime) & (df1['TIMESTAMP'] <= endtime)]
    df2 = df2[(df2['TIMESTAMP'] >= starttime) & (df2['TIMESTAMP'] <= endtime)]
    data = pd.merge(df1[['TIMESTAMP', 'ENMO']], df2[['TIMESTAMP', 'ENMO']], on='TIMESTAMP', suffixes=('_df1', '_df2'))
    data['ENMO_DIFF'] = abs(data['ENMO_df1'] - data['ENMO_df2'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='TIMESTAMP', y='ENMO_DIFF')
    plt.xlabel('ENMO_DIFF')
    plt.ylabel('Time')
    plt.title('ENMO Difference per Minute')
    plt.show()

    print(data['ENMO_DIFF'].describe())