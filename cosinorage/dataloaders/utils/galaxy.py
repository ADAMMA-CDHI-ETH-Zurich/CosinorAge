import pandas as pd
import os
from claid.data_collection.load.load_sensor_data import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def read_galaxy_data(gw_file_dir: str, meta_dict: dict, verbose: bool = False):

    data = pd.DataFrame()

    n_files = 0
    for day_dir in os.listdir(gw_file_dir):
        if os.path.isdir(gw_file_dir + day_dir):
            for file in os.listdir(gw_file_dir + day_dir):
                # only consider binary files
                if file.endswith(".binary") and file.startswith("acceleration_data"):
                    _temp = acceleration_data_to_dataframe(load_acceleration_data(gw_file_dir + day_dir + "/" + file))
                    data = pd.concat([data, _temp])
                    n_files += 1

    if verbose:
        print(f"Read {n_files} files from {gw_file_dir}")

    data = data.rename(columns={'unix_timestamp_in_ms': 'TIMESTAMP', 'acceleration_x': 'X', 'acceleration_y': 'Y', 'acceleration_z': 'Z'})
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='ms')
    data.set_index('TIMESTAMP', inplace=True)
    data.drop(columns=['effective_time_frame', 'sensor_body_location'], inplace=True)

    #min_val = data[['X', 'Y', 'Z']].min().min()
    #max_val = data[['X', 'Y', 'Z']].max().max()
    #data[['X', 'Y', 'Z']] = 6*(data[['X', 'Y', 'Z']] - min_val) / (max_val - min_val)
    data = data.fillna(0)
    data.sort_index()

    # compute raw data frequency
    meta_dict['raw_data_frequency'] = int(1 / (data.index[1] - data.index[0]).total_seconds())
    meta_dict['raw_data_type'] = 'accelerometer'
    meta_dict['raw_data_unit'] = 'm/s^2'

    return data

def acceleration_data_to_dataframe(data):
    rows = []
    for sample in data.samples:
        rows.append({
            'acceleration_x': sample.acceleration_x,
            'acceleration_y': sample.acceleration_y,
            'acceleration_z': sample.acceleration_z,
            'sensor_body_location': sample.sensor_body_location,
            'unix_timestamp_in_ms': sample.unix_timestamp_in_ms,
            'effective_time_frame': sample.effective_time_frame
        })

    return pd.DataFrame(rows)