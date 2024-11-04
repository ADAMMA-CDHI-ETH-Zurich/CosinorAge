import numpy as np

def calculate_enmo(data_all):
    """
    Calculate the ENMO metric from raw accelerometer data.
    """
    acc_vectors = data_all[['X', 'Y', 'Z']].values
    enmo = np.linalg.norm(acc_vectors, axis=1) - 1
    data_all['ENMO'] = np.maximum(enmo, 0)
    return data_all


def calculate_minute_level_enmo(data_all, sample_rate=80):
    """
    Resample to minute-level ENMO.
    """
    data_all.set_index('TIMESTAMP', inplace=True)
    minute_enmo = data_all['ENMO'].resample('min').mean().reset_index()
    return minute_enmo