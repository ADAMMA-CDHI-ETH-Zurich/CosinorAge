import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from .utils.calc_enmo import calculate_enmo, calculate_minute_level_enmo
from .utils.filtering import filter_incomplete_days
from .utils.smartwatch import read_smartwatch_data, preprocess_smartwatch_data
from .utils.ukbiobank import read_ukbiobank_data


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
        meta_dic (dict): A dictionary storing metadata.
        enmo_df (pd.DataFrame): A DataFrame storing minute-level ENMO values.
    """

    def __init__(self, datasource: str, input_path: str, preprocess: bool = True):
        """
        Initializes an empty DataLoader instance with an empty DataFrame
        for storing minute-level ENMO values.

        Args:
            datasource (str): The source of the data ('smartwatch', 'nhanes', or 'uk-biobank').
            input_path (str): The path to the input data.
            preprocess (bool): Whether to preprocess the data.
        """
        self.datasource = datasource
        self.input_path = input_path

        # check if filepaths are valid w.r.t. the datasource
        if datasource == 'smartwatch':
            if not os.path.isdir(input_path):
                raise ValueError("The input path should be a directory path")

        elif datasource == 'nhanes':
            if not os.path.isfile(input_path):
                raise ValueError("The input path should be a file path")

        elif datasource == 'uk-biobank':
            if not os.path.isfile(input_path):
                raise ValueError("The input path should be a file path")

        else:
            raise ValueError("The datasource should be either 'smartwatch', 'nhanes' or 'uk-biobank'")

        self.preprocess = preprocess

        if datasource in ['nhanes', 'smartwatch']:
            self.acc_df = None
            self.acc_freq = None

        self.meta_dic = {}
        self.enmo_df = None

    def load_data(self, verbose: bool = False, autocalib_max_iter: int = 1000, autocalib_tol: float = 1e-10):
        """
        Load data into the DataLoader instance.

        This method is intended to be implemented by subclasses. It should
        load data and store the minute-level ENMO values in `self.enmo`.

        Args:
            verbose (bool): Whether to print detailed information during loading.
            autocalib_max_iter (int): Maximum iterations for auto-calibration.
            autocalib_tol (float): Tolerance for auto-calibration.
        """
        if self.datasource == 'smartwatch':
            # load accelerometer data from csv files into a DataFrame
            self.acc_df, self.acc_freq = read_smartwatch_data(self.input_path)
            if verbose:
                print(f"Loaded {self.acc_df.shape[0]} accelerometer data records from {self.input_path}")
                print(f"The frequency of the accelerometer data is {self.acc_freq}Hz")

            # filter out incomplete days
            n = self.acc_df.shape[0]
            self.acc_df = filter_incomplete_days(self.acc_df, self.acc_freq)
            if verbose:
                print(f"Filtered out {n - self.acc_df.shape[0]} accelerometer records due to incomplete daily coverage")

            # if not data left, return empty DataFrame
            if self.acc_df.empty:
                self.enmo_df = pd.DataFrame()
                return

            # conduct preprocessing if required
            if self.preprocess:
                self.acc_df[['X', 'Y', 'Z', 'wear']] = preprocess_smartwatch_data(self.acc_df[['X', 'Y', 'Z']], self.acc_freq, self.meta_dic, max_iter=autocalib_max_iter, tol=autocalib_tol, verbose=verbose)
                if verbose:
                    print(f"Preprocessed accelerometer data")

            # calculate ENMO values at original frequency
            self.acc_df['ENMO'] = calculate_enmo(self.acc_df)
            if verbose:
                print(f"Calculated ENMO for {self.acc_df['ENMO'].shape[0]} accelerometer records")

            # aggregate ENMO values at the minute level
            self.enmo_df = calculate_minute_level_enmo(self.acc_df)
            if verbose:
                print(f"Aggregated ENMO values at the minute level leading to {self.enmo_df.shape[0]} records")

        elif self.datasource == 'uk-biobank':
            self.enmo_df = read_ukbiobank_data(self.input_path, source='uk-biobank')
            if verbose:
                print(f"Loaded {self.enmo_df.shape[0]} minute-level ENMO records from {self.input_path}")

            self.enmo_df = filter_incomplete_days(self.enmo_df, data_freq=1 / 60)
            if verbose:
                print(f"Filtered out {self.enmo_df.shape[0] - self.enmo_df.shape[0]} minute-level ENMO records due to incomplete daily coverage")

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

    def get_acc_data(self):
        """
        Retrieve the accelerometer data.

        Returns:
            pd.DataFrame: A DataFrame containing the accelerometer data.
        """
        if self.acc_df is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.acc_df

    def get_meta_data(self):
        """
        Retrieve the metadata.

        Returns:
            dict: A dictionary containing the metadata.
        """
        return self.meta_dic

    def plot_orig_enmo(self, resample: str = '15min', wear: bool = True):
        """
        Plot the original ENMO values resampled at a specified interval.

        Args:
            resample (str): The resampling interval (default is '15min').
            wear (bool): Whether to add color bands for wear and non-wear periods (default is True).
        """
        _data = self.acc_df.resample('15min').mean().reset_index(inplace=False)

        plt.figure(figsize=(12, 6))
        plt.plot(_data['TIMESTAMP'], _data['ENMO'], label='ENMO', color='black')

        if wear:
            # Add color bands for wear and non-wear periods
            for i in range(len(_data) - 1):
                start_time = _data['TIMESTAMP'].iloc[i]
                end_time = _data['TIMESTAMP'].iloc[i + 1]
                color = 'green' if _data['wear'].iloc[i] == 1 else 'red'
                plt.axvspan(start_time, end_time, color=color, alpha=0.3)

        plt.show()

    def plot_enmo(self):
        """
        Plot minute-level ENMO values.

        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.enmo_df, x=self.enmo_df.index, y='ENMO')
        plt.xlabel('Time')
        plt.ylabel('ENMO')
        plt.title('ENMO per Minute')
        plt.xticks(rotation=45)
        plt.show()