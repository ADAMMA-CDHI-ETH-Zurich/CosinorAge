###########################################################################
# Copyright (C) 2024 ETH Zurich
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
from .utils.ukb import read_ukb_data, filter_ukb_data, resample_ukb_data
from .utils.nhanes import read_nhanes_data, filter_nhanes_data, resample_nhanes_data
from .utils.galaxy import read_galaxy_data, filter_galaxy_data, resample_galaxy_data, preprocess_galaxy_data


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


class DataHandler:
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
        sf_data (pd.DataFrame): A DataFrame storing accelerometer data.
        acc_freq (int): The frequency of the accelerometer data.
        meta_dict (dict): A dictionary storing metadata.
        ml_data (pd.DataFrame): A DataFrame storing minute-level ENMO values.
    """

    def __init__(self):
        """
        Initializes an empty DataHandler instance with an empty DataFrame
        for storing minute-level ENMO values.

        Args:
            datasource (str): The source of the data ('smartwatch', 'nhanes', or 'uk-biobank').
            input_path (str): The path to the input data.
            preprocess (bool): Whether to preprocess the data.
        """
        self.raw_data = None
        self.sf_data = None
        self.ml_data = None

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
        if self.ml_data is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        self.ml_data.to_csv(output_path, index=False)

    def get_raw_data(self):
        """
        Retrieve the raw data.

        Returns:
            pd.DataFrame: A DataFrame containing the raw data.
        """
        return self.raw_data

    def get_sf_data(self):
        """
        Retrieve the accelerometer data.

        Returns:
            pd.DataFrame: A DataFrame containing the accelerometer data.
        """
        if self.sf_data is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.sf_data

    def get_ml_data(self):
        """
        Retrieve the minute-level ENMO values.

        Returns:
            pd.DataFrame: A DataFrame containing the minute-level ENMO values.
        """
        if self.ml_data is None:
            raise ValueError("Data has not been loaded. Please call `load_data()` first.")

        return self.ml_data

    def get_meta_data(self):
        """
        Retrieve the metadata.

        Returns:
            dict: A dictionary containing the metadata.
        """
        return self.meta_dict

    
