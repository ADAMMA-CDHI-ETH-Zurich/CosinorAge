import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import welch


from .utils.calc_enmo import calculate_enmo, calculate_minute_level_enmo
from .utils.filtering import filter_incomplete_days, filter_consecutive_days
from .utils.smartwatch import read_smartwatch_data, preprocess_smartwatch_data
from .utils.ukb import read_ukbiobank_data
from .utils.nhanes import read_nhanes_data
from .utils.galaxy import read_galaxy_data
def clock(func):
    """
    A decorator that prints the execution time of the decorated function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result

    return inner


class DataLoader:
    """
    A base class for data loaders that process and store ENMO data at the
    minute level.

    This class provides a common interface for data loaders with methods to load
    data, retrieve processed ENMO values, and save data. The `load_data` and
    `save_data` methods are intended to be overridden by subclasses.

    Attributes:
        datasource (str): The source of the data ('smartwatch', 'nhanes', or 'uk-biobank').
        input_path (str): The path to the input data.
        preprocess (bool): Whether to preprocess the data.
        acc_df (pd.DataFrame): A DataFrame storing accelerometer data.
        acc_freq (int): The frequency of the accelerometer data.
        meta_dict (dict): A dictionary storing metadata.
        enmo_df (pd.DataFrame): A DataFrame storing minute-level ENMO values.
    """

    def __init__(self):
        """
        Initializes an empty DataLoader instance with an empty DataFrame
        for storing minute-level ENMO values.

        Args:
            datasource (str): The source of the data ('smartwatch', 'nhanes', or 'uk-biobank').
            input_path (str): The path to the input data.
            preprocess (bool): Whether to preprocess the data.
        """
        self.enmo_df = None
        self.meta_dict = {}

    def load_data(self, verbose: bool = False):
        raise NotImplementedError("The load_data method should be implemented by subclasses")

    def save_data(self, output_path: str):
        """
        Save minute-level ENMO data to a specified output path.

        This method is intended to be implemented by subclasses, specifying
        the format and structure for saving data.

        Args:
            output_path (str): The file path where the minute-level ENMO data
                will be saved.
        """
        if self.enmo_df is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        self.enmo_df.to_csv(output_path, index=False)

    def get_enmo_data(self):
        """
        Retrieve the minute-level ENMO values.

        Returns:
            pd.DataFrame: A DataFrame containing the minute-level ENMO values.
        """
        if self.enmo_df is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.enmo_df

    def get_meta_data(self):
        """
        Retrieve the metadata.

        Returns:
            dict: A dictionary containing the metadata.
        """
        return self.meta_dict


class AccelerometerDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.acc_df = None

    def get_acc_data(self):
        """
        Retrieve the accelerometer data.

        Returns:
            pd.DataFrame: A DataFrame containing the accelerometer data.
        """
        if self.acc_df is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.acc_df

class ENMODDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
    

class NHANESDataLoader(ENMODDataLoader):
    def __init__(self, nhanes_file_dir: str, person_id: str = None):
        super().__init__()

        if not os.path.isdir(nhanes_file_dir):
            raise ValueError("The input path should be a directory path")

        self.nhanes_file_dir = nhanes_file_dir

        self.person_id = person_id

        self.meta_dict['datasource'] = 'nhanes'
    
    @clock
    def load_data(self, verbose: bool = False):
        if self.person_id is None:
                raise ValueError("The person_id is required for nhanes data")

        self.enmo_df = read_nhanes_data(self.nhanes_file_dir, meta_dict=self.meta_dict, verbose=verbose, person_id=self.person_id)
        if verbose:
            print(f"Loaded {self.enmo_df.shape[0]} minute-level ENMO records from {self.nhanes_file_dir}")
        
        old_n = self.enmo_df.shape[0]
        self.enmo_df = filter_incomplete_days(self.enmo_df, data_freq=self.meta_dict['raw_data_frequency'])
        if verbose:
            print(f"Filtered out {old_n - self.enmo_df.shape[0]} minute-level ENMO records due to incomplete daily coverage")

        self.enmo_df.index = pd.to_datetime(self.enmo_df.index)
        
        old_n = self.enmo_df.shape[0]
        self.enmo_df = filter_consecutive_days(self.enmo_df)
        if verbose:
            print(f"Filtered out {old_n - self.enmo_df.shape[0]} minute-level ENMO records due to filtering for longest consecutive sequence of days")

        self.meta_dict['n_days'] = self.enmo_df.index.date.nunique()

class UKBDataLoader(AccelerometerDataLoader):
    def __init__(self, qa_file_path: str, ukb_file_dir: str, person_id: str = None):
        super().__init__()

        if not os.path.isfile(qa_file_path):
            raise ValueError("The QA file path should be a file path")
        if not os.path.isdir(ukb_file_dir):
            raise ValueError("The UKB file directory should be a directory path")

        self.qa_file_path = qa_file_path
        self.ukb_file_dir = ukb_file_dir

        self.person_id = person_id

        self.meta_dict['datasource'] = 'uk-biobank'

    def load_data(self, verbose: bool = False):
        pass

class GalaxyDataLoader(AccelerometerDataLoader):
    def __init__(self, gw_file_dir: str, preprocess: bool = True, preprocess_args: dict = {}):
        super().__init__()

        if not os.path.isdir(gw_file_dir):
            raise ValueError("The Galaxy Watch file directory should be a directory path")

        self.gw_file_dir = gw_file_dir

        self.raw_data = None

        self.preprocess = preprocess
        self.preprocess_args = preprocess_args

        self.meta_dict['datasource'] = 'samsung galaxy watch'
    
    @clock
    def load_data(self, verbose: bool = False):
        # load raw data
        self.raw_data = read_galaxy_data(self.gw_file_dir, meta_dict=self.meta_dict, verbose=verbose)
        if verbose:
            print(f"Loaded {self.raw_data.shape[0]} accelerometer data records from {self.gw_file_dir}")
        self.meta_dict['raw_n_timesteps'] = self.raw_data.shape[0]
        self.meta_dict['raw_n_days'] = len(np.unique(self.raw_data.index.date))
        self.meta_dict['raw_start_datetime'] = self.raw_data.index.min()
        self.meta_dict['raw_end_datetime'] = self.raw_data.index.max()
        self.meta_dict['raw_frequency'] = 'irregular (~25Hz)'
        self.meta_dict['raw_datatype'] = 'accelerometer'
        self.meta_dict['raw_unit'] = ''

        # filter out first and last day
        n_old = self.raw_data.shape[0]
        self.acc_df = self.raw_data.loc[(self.raw_data.index.date != self.raw_data.index.date.min()) & (self.raw_data.index.date != self.raw_data.index.date.max())]
        if verbose:
            print(f"Filtered out {self.raw_data.shape[0] - self.acc_df.shape[0]}/{self.raw_data.shape[0]} accelerometer records due to filtering out first and last day")

        # filter out sparse days
        n_old = self.acc_df.shape[0]
        self.acc_df = filter_incomplete_days(self.acc_df, data_freq=25, expected_points_per_day=2000000)
        if verbose:
            print(f"Filtered out {n_old - self.acc_df.shape[0]}/{n_old} accelerometer records due to incomplete daily coverage")

        # filter for longest consecutive sequence of days
        old_n = self.acc_df.shape[0]
        self.acc_df = filter_consecutive_days(self.acc_df)
        if verbose:
            print(f"Filtered out {old_n - self.acc_df.shape[0]}/{old_n} minute-level accelerometer records due to filtering for longest consecutive sequence of days")

        # bring data so consistent frequency of 25Hz 
        n_old = self.acc_df.shape[0]
        self.acc_df = self.acc_df.resample('40ms').interpolate(method='linear').bfill()
        if verbose:
            print(f"Resampled {n_old} to {self.acc_df.shape[0]} timestamps")
        self.meta_dict['resampled_n_timestamps'] = self.acc_df.shape[0]
        self.meta_dict['resampled_n_days'] = len(np.unique(self.acc_df.index.date))
        self.meta_dict['resampled_start_datetime'] = self.acc_df.index.min()
        self.meta_dict['resampled_end_datetime'] = self.acc_df.index.max()
        self.meta_dict['resampled_frequency'] = '25Hz'
        self.meta_dict['resampled_datatype'] = 'accelerometer'
        self.meta_dict['resampled_unit'] = ''

        if self.preprocess:
            self.acc_df[['X_raw', 'Y_raw', 'Z_raw']] = self.acc_df[['X', 'Y', 'Z']]
            self.acc_df[['X', 'Y', 'Z', 'wear']] = preprocess_smartwatch_data(self.acc_df[['X', 'Y', 'Z']], 25, self.meta_dict, preprocess_args=self.preprocess_args, verbose=verbose)
            if verbose:
                print(f"Preprocessed accelerometer data")

        self.meta_dict['preprocessed_n_timesteps'] = self.acc_df.shape[0]
        self.meta_dict['preprocessed_n_days'] = len(np.unique(self.acc_df.index.date))
        self.meta_dict['preprocessed_start_datetime'] = self.acc_df.index.min()
        self.meta_dict['preprocessed_end_datetime'] = self.acc_df.index.max()
        self.meta_dict['preprocessed_frequency'] = '25Hz'
        self.meta_dict['preprocessed_datatype'] = 'accelerometer'
        self.meta_dict['preprocessed_unit'] = ''

        # calculate ENMO values at original frequency
        self.acc_df['ENMO'] = calculate_enmo(self.acc_df)
        if verbose:
            print(f"Calculated ENMO for {self.acc_df['ENMO'].shape[0]} accelerometer records")

        # aggregate ENMO values at the minute level in mg
        self.enmo_df = calculate_minute_level_enmo(self.acc_df, 25)
        self.enmo_df['ENMO'] = self.enmo_df['ENMO']-self.enmo_df['ENMO'].min()
        self.enmo_df.index = pd.to_datetime(self.enmo_df.index)
        if verbose:
            print(f"Aggregated ENMO values at the minute level leading to {self.enmo_df.shape[0]} records")

        self.meta_dict['minute-level_n_timesteps'] = self.enmo_df.shape[0]
        self.meta_dict['minute-level_n_days'] = len(np.unique(self.enmo_df.index.date))
        self.meta_dict['minute-level_start_datetime'] = self.enmo_df.index.min()
        self.meta_dict['minute-level_end_datetime'] = self.enmo_df.index.max()
        self.meta_dict['minute-level_frequency'] = 'minute-level'
        self.meta_dict['minute-level_datatype'] = 'enmo'
        self.meta_dict['minute-level_unit'] = ''


class SmartwatchDataLoader(AccelerometerDataLoader):

    def __init__(self, smartwatch_file_dir: str, preprocess: bool = True, preprocess_args: dict = {}):
        super().__init__()
        
        if not os.path.isdir(smartwatch_file_dir):
            raise ValueError("The smartwatch file directory should be a directory path")
        
        self.smartwatch_file_dir = smartwatch_file_dir

        self.preprocess = preprocess
        self.preprocess_args = preprocess_args

        self.meta_dict['datasource'] = 'smartwatch'

    @clock
    def load_data(self, verbose: bool = False):
        # load accelerometer data from csv files into a DataFrame
        self.acc_df = read_smartwatch_data(self.smartwatch_file_dir, meta_dict=self.meta_dict)
        if verbose:
            print(f"Loaded {self.acc_df.shape[0]} accelerometer data records from {self.smartwatch_file_dir}")

        # filter out incomplete days
        n_old = self.acc_df.shape[0]
        self.acc_df = filter_incomplete_days(self.acc_df, self.meta_dict['raw_data_frequency'])
        if verbose:
            print(f"Filtered out {n_old - self.acc_df.shape[0]} accelerometer records due to incomplete daily coverage")

        # if not data left, return
        if self.acc_df.empty:
            self.enmo_df = pd.DataFrame()
            return

        # conduct preprocessing if required
        if self.preprocess:
            self.acc_df[['X_raw', 'Y_raw', 'Z_raw']] = self.acc_df[['X', 'Y', 'Z']]
            self.acc_df[['X', 'Y', 'Z', 'wear']] = preprocess_smartwatch_data(self.acc_df[['X', 'Y', 'Z']], self.meta_dict['raw_data_frequency'], self.meta_dict, preprocess_args=self.preprocess_args, verbose=verbose)
            if verbose:
                print(f"Preprocessed accelerometer data")

        # calculate ENMO values at original frequency
        self.acc_df['ENMO'] = calculate_enmo(self.acc_df)*1000
        if verbose:
            print(f"Calculated ENMO for {self.acc_df['ENMO'].shape[0]} accelerometer records")

        # aggregate ENMO values at the minute level in mg
        self.enmo_df = calculate_minute_level_enmo(self.acc_df, self.meta_dict['raw_data_frequency'])
        self.enmo_df['ENMO'] = self.enmo_df['ENMO']
        self.enmo_df.index = pd.to_datetime(self.enmo_df.index)
        if verbose:
            print(f"Aggregated ENMO values at the minute level leading to {self.enmo_df.shape[0]} records")

        self.meta_dict['n_days'] = len(np.unique(self.enmo_df.index.date))
