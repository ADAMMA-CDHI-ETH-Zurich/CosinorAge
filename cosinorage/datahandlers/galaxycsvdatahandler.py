###########################################################################
# Copyright (C) 2025 ETH Zurich
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

import os

from .utils.calc_enmo import calculate_minute_level_enmo
from .utils.galaxy_csv import read_galaxy_csv_data, filter_galaxy_csv_data, resample_galaxy_csv_data, preprocess_galaxy_csv_data
from .datahandler import DataHandler, clock


class GalaxyCSVDataHandler(DataHandler):
    """
    Data handler for Samsung Galaxy Watch accelerometer data in CSV format.

    This class handles loading, filtering, and processing of Galaxy Watch accelerometer data in CSV format.

    Args:
        galaxy_file_path (str): Path to the Galaxy Watch data file.
        preprocess (bool, optional): Whether to preprocess the data. Defaults to True.
        preprocess_args (dict, optional): Arguments for preprocessing. Defaults to {}.
        verbose (bool, optional): Whether to print processing information. Defaults to False.

    Attributes:
        galaxy_file_path (str): Path to the Galaxy Watch data file.
        preprocess (bool): Whether to preprocess the data.
        preprocess_args (dict): Arguments for preprocessing.
    """

    def __init__(self, 
                 galaxy_file_path: str, 
                 preprocess_args: dict = {}, 
                 verbose: bool = False):

        super().__init__()

        if not os.path.isfile(galaxy_file_path):
            raise ValueError("The Galaxy Watch file path should be a file path")

        self.galaxy_file_path = galaxy_file_path
        self.preprocess_args = preprocess_args

        self.meta_dict['datasource'] = 'samsung galaxy smartwatch'

        self.__load_data(verbose=verbose)
    
    @clock
    def __load_data(self, 
                    verbose: bool = False):
        """
        Internal method to load and process Galaxy Watch data.

        Args:
            verbose (bool, optional): Whether to print processing information. Defaults to False.
        """

        self.raw_data = read_galaxy_csv_data(self.galaxy_file_path, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = filter_galaxy_csv_data(self.raw_data, meta_dict=self.meta_dict, verbose=verbose, preprocess_args=self.preprocess_args)
        self.sf_data = resample_galaxy_csv_data(self.sf_data, meta_dict=self.meta_dict, verbose=verbose)
        self.sf_data = preprocess_galaxy_csv_data(self.sf_data, preprocess_args=self.preprocess_args, meta_dict=self.meta_dict, verbose=verbose)
        self.ml_data = calculate_minute_level_enmo(self.sf_data, sf=1/60, verbose=verbose)
