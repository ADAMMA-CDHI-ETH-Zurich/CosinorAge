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

from .datahandler import DataHandler, clock

    
class NHANESDataHandler(DataHandler):
    """
    Data loader for NHANES accelerometer data.

    This class handles loading, filtering, and processing of NHANES accelerometer data.

    Args:
        nhanes_file_dir (str): Directory path containing NHANES data files.
        person_id (str, optional): Specific person ID to load. Defaults to None.
        verbose (bool, optional): Whether to print processing information. Defaults to False.

    Attributes:
        nhanes_file_dir (str): Directory containing NHANES data files.
        person_id (str): ID of the person whose data is being loaded.
    """

    def __init__(self, nhanes_file_dir: str, person_id: str = None, verbose: bool = False):
        super().__init__()

        if not os.path.isdir(nhanes_file_dir):
            raise ValueError("The input path should be a directory path")

        self.nhanes_file_dir = nhanes_file_dir
        self.person_id = person_id

        self.meta_dict['datasource'] = 'nhanes'

        self.__load_data(verbose=verbose)
    
    @clock
    def __load_data(self, verbose: bool = False):
        """
        Internal method to load and process NHANES data.

        Args:
            verbose (bool, optional): Whether to print processing information. Defaults to False.
        """

        self.raw_data = read_nhanes_data(self.nhanes_file_dir, person_id=self.person_id, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = filter_nhanes_data(self.raw_data, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = resample_nhanes_data(self.sf_data, meta_dict=self.meta_dict, verbose=verbose)
        self.ml_data = self.sf_data

