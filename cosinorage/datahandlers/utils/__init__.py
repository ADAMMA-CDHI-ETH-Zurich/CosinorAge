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

# Import functions from various utility modules
from .calc_enmo import calculate_enmo, calculate_minute_level_enmo
from .filtering import filter_incomplete_days, filter_consecutive_days, largest_consecutive_sequence
from .frequency_detection import detect_frequency_from_timestamps
from .galaxy_binary import (
    acceleration_data_to_dataframe,
    filter_galaxy_binary_data,
    preprocess_galaxy_binary_data,
    read_galaxy_binary_data,
    resample_galaxy_binary_data,
)
from .galaxy_csv import (
    filter_galaxy_csv_data,
    preprocess_galaxy_csv_data,
    read_galaxy_csv_data,
    resample_galaxy_csv_data,
)
from .generic import read_generic_xD_data, filter_generic_data, resample_generic_data, preprocess_generic_data
from .nhanes import (
    calculate_measure_time,
    clean_data,
    filter_and_preprocess_nhanes_data,
    read_nhanes_data,
    remove_bytes,
    resample_nhanes_data,
)
from .ukb import filter_ukb_data, read_ukb_data, resample_ukb_data
from .visualization import plot_enmo, plot_orig_enmo, plot_orig_enmo_freq

__all__ = [
    # calc_enmo
    "calculate_enmo",
    "calculate_minute_level_enmo",
    # filtering
    "filter_incomplete_days",
    "filter_consecutive_days",
    "largest_consecutive_sequence",
    # frequency_detection
    "detect_frequency_from_timestamps",
    # galaxy_binary
    "acceleration_data_to_dataframe",
    "filter_galaxy_binary_data",
    "preprocess_galaxy_binary_data",
    "read_galaxy_binary_data",
    "resample_galaxy_binary_data",
    # galaxy_csv
    "filter_galaxy_csv_data",
    "preprocess_galaxy_csv_data",
    "read_galaxy_csv_data",
    "resample_galaxy_csv_data",
    # generic
    "read_generic_xD_data",
    "filter_generic_data",
    "resample_generic_data",
    "preprocess_generic_data",
    # nhanes
    "calculate_measure_time",
    "clean_data",
    "filter_and_preprocess_nhanes_data",
    "read_nhanes_data",
    "remove_bytes",
    "resample_nhanes_data",
    # ukb
    "filter_ukb_data",
    "read_ukb_data",
    "resample_ukb_data",
    # visualization
    "plot_enmo",
    "plot_orig_enmo",
    "plot_orig_enmo_freq",
]

