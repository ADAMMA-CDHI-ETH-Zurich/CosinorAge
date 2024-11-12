import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from .utils.calc_enmo import calculate_enmo, calculate_minute_level_enmo
from .utils.read_csv import read_acc_csvs, read_enmo_csv, filter_incomplete_days


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

    def load_data(self):
        """
        Load data into the DataLoader instance.

        This method is intended to be implemented by subclasses. It should
        load data
        and store the minute-level ENMO values in `self.enmo`.
        """

        if self.datasource == 'smartwatch':
            self.acc_df, self.acc_freq = read_acc_csvs(self.input_path)
            print(f"Loaded {self.acc_df.shape[0]} accelerometer data records from {self.input_path}")
            print(f"The frequency of the accelerometer data is {self.acc_freq}Hz")

            n = self.acc_df.shape[0]
            self.acc_df = filter_incomplete_days(self.acc_df, self.acc_freq)
            print(f"Filtered out {n - self.acc_df.shape[0]} accelerometer records due to incomplete daily coverage")

            if self.acc_df.empty:
                self.enmo_df = pd.DataFrame()
                return

            _enmo = calculate_enmo(self.acc_df)
            print(f"Calculated ENMO for {_enmo.shape[0]} accelerometer records")

            self.enmo_df = calculate_minute_level_enmo(_enmo).reset_index(drop=True)
            print(f"Aggregated ENMO values at the minute level leading to {self.enmo_df.shape[0]} records")

            self.enmo_df.set_index('TIMESTAMP', inplace=True)

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


class AccelerometerDataLoader(DataLoader):
    """
    A data loader for processing accelerometer data. This class loads,
    processes, and saves accelerometer data, calculating ENMO
    (Euclidean Norm Minus One) values at the minute level.

    Attributes:
        input_dir_path (str): Path to the directory containing input CSV files.

        acc (pd.DataFrame): DataFrame containing raw and processed
            accelerometer data.

        enmo (pd.DataFrame): DataFrame containing ENMO values aggregated at
            the minute level.
    """

    def __init__(self, input_dir_path: str):
        """
        Initializes the AccelerometerDataLoader with the path to the input
        data directory.

        Args:
            input_dir_path (str): The path to the directory containing input
            CSV files.
        """
        super().__init__()
        self.input_dir_path = input_dir_path

        self.acc_df = None
        self.acc_freq = None

        self.acc_fil_df = None

        self.enmo_df = None

    @clock
    def load_data(self):
        """
        Loads and processes accelerometer data from CSV files in the
        specified directory.
        This method performs several transformations, including timestamp
        conversion,
        ENMO calculation, and filtering of incomplete days. It then
        aggregates ENMO
        values at the minute level and stores the result in `self.enmo_df`.

        Processing steps include:
            1. Concatenating CSV files from the input directory.
            2. Converting timestamps to POSIX format.
            3. Calculating the ENMO metric.
            4. Filtering out incomplete days.
            5. Aggregating ENMO values at the minute level.

        Returns:
            None
        """

        if (self.enmo_df is not None or self.enmo_minute_fil_df is not None or
                self.acc_df is not None or self.acc_fil_df is not None):
            raise ValueError(
                "Data has already been loaded. Please create a new instance "
                "to load new data.")

        self.acc_df, self.acc_freq = read_acc_csvs(self.input_dir_path)
        print(
            f"Loaded {self.acc_df.shape[0]} accelerometer data records from "
            f"{self.input_dir_path}")
        print(f"The frequency of the accelerometer data is {self.acc_freq}Hz")

        self.acc_fil_df = filter_incomplete_days(self.acc_df, self.acc_freq)
        print(
            f"Filtered out {self.acc_df.shape[0] - self.acc_fil_df.shape[0]} "
            f"accelerometer records due to incomplete daily coverage")

        if self.acc_fil_df.empty:
            self.enmo_df = pd.DataFrame()
            self.enmo_minute_fil_df = pd.DataFrame()
            return

        self.enmo_df = calculate_enmo(self.acc_fil_df)
        print(
            f"Calculated ENMO for {self.enmo_df.shape[0]} accelerometer "
            f"records")

        self.enmo_minute_fil_df = calculate_minute_level_enmo(
            self.enmo_df).reset_index(drop=True)
        print(
            f"Aggregated ENMO values at the minute level leading to "
            f"{self.enmo_minute_fil_df.shape[0]} records")

        self.enmo_minute_fil_df.set_index('TIMESTAMP', inplace=True)

    @clock
    def save_data(self, output_file_path: str):
        """
        Saves the processed minute-level ENMO data to a CSV file.

        Args:
            output_file_path (str): The file path where the minute-level ENMO
            data will be saved.

        Returns:
            None
        """

        if self.enmo_df is None:
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first.")

        self.enmo_df.to_csv(output_file_path, index=False)


class ENMODataLoader(DataLoader):
    """
    A data loader for processing ENMO data from a single CSV file. This class
    loads, processes, and saves ENMO (Euclidean Norm Minus One) values at the
    minute level.

    Attributes:
        input_file_path (str): Path to the input CSV file containing ENMO data.
        enmo (pd.DataFrame): DataFrame containing processed ENMO values.
    """

    def __init__(self, input_file_path: str):
        """
        Initializes the ENMODataLoader with the path to the input data file.

        Args:
            input_file_path (str): The path to the CSV file containing ENMO
            data.
        """
        super().__init__()
        self.input_file_path = input_file_path

        self.enmo_minute_df = None

        self.enmo_freq = 1 / 60

    @clock
    def load_data(self):
        """
        Loads and processes ENMO data from the specified CSV file. This method
        performs several transformations, including timestamp conversion, data
        renaming, and filtering of incomplete days. It then stores the processed
        data in `self.enmo_df`.

        Processing steps include:
            1. Loading data from a CSV file.
            2. Selecting 'time' and 'ENMO_t' columns.
            3. Converting timestamps to POSIX format.
            4. Renaming 'ENMO_t' to 'ENMO'.
            5. Dropping unnecessary columns.
            6. Filtering out incomplete days.

        Returns:
            None
        """

        if self.enmo_minute_df is not None:
            raise ValueError(
                "Data has already been loaded. Please create a new instance "
                "to load new data.")

        # Load and preprocess data
        self.enmo_minute_df = read_enmo_csv(self.input_file_path,
                                            source='uk-biobank')
        print(
            f"Loaded {self.enmo_minute_df.shape[0]} minute-level ENMO records "
            f"from {self.input_file_path}")

        # Filter for complete days and reset index
        self.enmo_minute_fil_df = filter_incomplete_days(self.enmo_minute_df,
                                                         self.enmo_freq)
        print(
            f"Filtered out "
            f"{self.enmo_minute_df.shape[0] - self.enmo_minute_fil_df.shape[0]}"
            f" minute-level ENMO records due to incomplete daily coverage")

        self.enmo_minute_fil_df.set_index('TIMESTAMP', inplace=True)

    @clock
    def save_data(self, output_file_path: str):
        """
        Saves the processed minute-level ENMO data to a CSV file.

        Args:
            output_file_path (str): The file path where the minute-level ENMO
            data will be saved.

        Returns:
            None
        """

        if self.enmo_minute_fil_df is None:
            raise ValueError(
                "Data has not been loaded. Please call `load_data()` first.")

        self.enmo_minute_fil_df.to_csv(output_file_path, index=False)
