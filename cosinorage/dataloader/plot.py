import matplotlib.pyplot as plt
import seaborn as sns
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
    data = loader_1.get_enmo_per_minute().copy()
    enmo_diff = loader_1.get_enmo_per_minute()['ENMO'] - loader_2.get_enmo_per_minute()['ENMO']
    data['ENMO_DIFF'] = enmo_diff
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='TIMESTAMP', y='ENMO_DIFF')
    plt.xlabel('ENMO_DIFF')
    plt.ylabel('Time')
    plt.title('ENMO Difference per Minute')
    plt.show()