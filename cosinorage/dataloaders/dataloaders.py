import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from .utils.calc_enmo import calculate_enmo, calculate_minute_level_enmo
from .utils.read_csv import read_acc_csvs, read_enmo_csv, filter_incomplete_days
from .utils.smartwatch import preprocess_smartwatch_data


def clock(func):
    def inner(*args, **kwargs):
        import time
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
        enmo (pd.DataFrame): A DataFrame storing minute-level ENMO values.
    """

    def __init__(self, datasource: str, input_path: str, preprocess: bool = True):
        """
        Initializes an empty DataLoader instance with an empty DataFrame
        for storing minute-level ENMO values.
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

        self.enmo_df = None

    def load_data(self, verbose: bool = False):
        """
        Load data into the DataLoader instance.

        This method is intended to be implemented by subclasses. It should
        load data
        and store the minute-level ENMO values in `self.enmo`.
        """

        if self.datasource == 'smartwatch':
            self.acc_df, self.acc_freq = read_acc_csvs(self.input_path)
            if verbose:
                print(f"Loaded {self.acc_df.shape[0]} accelerometer data records from {self.input_path}")
                print(f"The frequency of the accelerometer data is {self.acc_freq}Hz")

            n = self.acc_df.shape[0]
            self.acc_df = filter_incomplete_days(self.acc_df, self.acc_freq)
            if verbose:
                print(f"Filtered out {n - self.acc_df.shape[0]} accelerometer records due to incomplete daily coverage")

            if self.acc_df.empty:
                self.enmo_df = pd.DataFrame()
                return

            if self.preprocess:
                self.acc_df[['X', 'Y', 'Z']] = preprocess_smartwatch_data(self.acc_df[['X', 'Y', 'Z']], self.acc_freq, max_iter=1)

                if verbose:
                    print(f"Preprocessed accelerometer data")

            print(len(self.acc_df.columns))
            self.acc_df['ENMO'] = calculate_enmo(self.acc_df)

            if verbose:
                print(f"Calculated ENMO for {self.acc_df['ENMO'].shape[0]} accelerometer records")

            self.enmo_df = calculate_minute_level_enmo(self.acc_df)

            if verbose:
                print(f"Aggregated ENMO values at the minute level leading to {self.enmo_df.shape[0]} records")

            #self.enmo_df.set_index('TIMESTAMP', inplace=True)

        elif self.datasource == 'uk-biobank':

            self.enmo_df = read_enmo_csv(self.input_path, source='nhanes')
            print(f"Loaded {self.enmo_df.shape[0]} minute-level ENMO records from {self.input_path}")

            self.enmo_df = filter_incomplete_days(self.enmo_df, data_freq=1 / 60)
            print(
                f"Filtered out {self.enmo_df.shape[0] - self.enmo_df.shape[0]} minute-level ENMO records due to incomplete daily coverage")

            self.enmo_df.set_index('TIMESTAMP', inplace=True)

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
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first.")

        self.enmo_df.to_csv(output_path, index=False)

    def get_enmo_data(self):
        """
        Retrieve the minute-level ENMO values.

        Returns:
            pd.DataFrame: A DataFrame containing the minute-level ENMO values.
        """

        if self.enmo_df is None:
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first.")

        return self.enmo_df

    def get_acc_data(self):
        """
        Retrieve the accelerometer data.

        Returns:
            pd.DataFrame: A DataFrame containing the accelerometer data.
        """

        if self.acc_df is None:
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first.")

        return self.acc_df

    def plot_enmo(self):
        """
        Plot minute-level ENMO values.

        Returns:
            None
        """

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.enmo_df, x='TIMESTAMP', y='ENMO')
        plt.xlabel('Time')
        plt.ylabel('ENMO')
        plt.title('ENMO per Minute')
        plt.xticks(rotation=45)
        plt.show()


