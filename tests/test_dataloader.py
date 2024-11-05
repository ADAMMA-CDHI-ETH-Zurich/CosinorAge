import numpy as np
import pandas as pd
from cosinorage.dataloader import AccelerometerDataLoader, ENMODataLoader

def test_AccelerometerDataLoader():
    acc_input_dir_path = "tests/data/62164/"
    acc_loader = AccelerometerDataLoader(input_dir_path=acc_input_dir_path)
    acc_loader.load_data()

    assert acc_loader.enmo_per_minute.shape[1] == 2, "AccelerometerDataLoader() ENMO Data Frame should have 2 columns"

    assert acc_loader.enmo_per_minute['TIMESTAMP'].min() == pd.Timestamp('2000-01-04 00:00:00'), "Minimum POSIX date does not match"
    assert acc_loader.enmo_per_minute['TIMESTAMP'].max() == pd.Timestamp('2000-01-08 23:59:00'), "Maximum POSIX date does not match"



    enmo_input_file_path = "tests/data/62164.csv"
    enmo_loader = ENMODataLoader(input_file_path=enmo_input_file_path)
    enmo_loader.load_data()

    assert np.linalg.norm(enmo_loader.enmo_per_minute['ENMO'] - acc_loader.enmo_per_minute['ENMO']) < 10e-14, "Minute-level ENMO values do not match"


def test_ENMODataLoader():
    enmo_input_file_path = "tests/data/62164.csv"
    enmo_loader = ENMODataLoader(input_file_path=enmo_input_file_path)
    enmo_loader.load_data()

    assert enmo_loader.enmo_per_minute.shape[1] == 2, "ENMODataLoader() ENMO Data Frame should have 2 columns"

    assert enmo_loader.enmo_per_minute['TIMESTAMP'].min() == pd.Timestamp('2000-01-04 00:00:00'), "Minimum POSIX date does not match"
    assert enmo_loader.enmo_per_minute['TIMESTAMP'].max() == pd.Timestamp('2000-01-08 23:59:00'), "Maximum POSIX date does not match"

