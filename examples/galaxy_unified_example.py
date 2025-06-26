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

"""
Example demonstrating the unified GalaxyDataHandler usage.

This example shows how to use the GalaxyDataHandler for both CSV and binary formats.
"""

from cosinorage.datahandlers import GalaxyDataHandler, plot_enmo
from cosinorage.features import WearableFeatures, dashboard


def example_csv_enmo():
    """Example using CSV format with ENMO data."""
    print("=== CSV ENMO Example ===")
    
    preprocess_args = {
        'required_daily_coverage': 0.4,
    }
    
    # Using the unified handler for CSV ENMO data with custom column names
    galaxy_handler = GalaxyDataHandler(
        galaxy_file_path='../data/smartwatch/sample1.csv', 
        data_format='csv', 
        data_type='enmo',  # Explicitly set (default would be 'enmo' for CSV)
        time_column='time',  # Custom time column name (default would be 'times')
        data_columns=['enmo_mg'],  # Custom data column name (default would be ['enmo_mg'])
        verbose=True, 
        preprocess_args=preprocess_args
    )
    
    print(f"Data format: {galaxy_handler.data_format}")
    print(f"Data type: {galaxy_handler.data_type}")
    print(f"Time column: {galaxy_handler.time_column}")
    print(f"Data columns: {galaxy_handler.data_columns}")
    
    # Get metadata
    meta_data = galaxy_handler.get_meta_data()
    print(f"Meta data: {meta_data}")
    
    return galaxy_handler


def example_csv_enmo_defaults():
    """Example using CSV format with ENMO data using default column names."""
    print("=== CSV ENMO Example (Defaults) ===")
    
    preprocess_args = {
        'required_daily_coverage': 0.4,
    }
    
    # Using the unified handler for CSV ENMO data with default column names
    galaxy_handler = GalaxyDataHandler(
        galaxy_file_path='../data/smartwatch/sample1.csv', 
        data_format='csv', 
        # data_type defaults to 'enmo' for CSV format
        # time_column defaults to 'times' for CSV ENMO data
        # data_columns defaults to ['enmo_mg'] for ENMO data
        verbose=True, 
        preprocess_args=preprocess_args
    )
    
    print(f"Data format: {galaxy_handler.data_format}")
    print(f"Data type: {galaxy_handler.data_type}")
    print(f"Time column: {galaxy_handler.time_column}")
    print(f"Data columns: {galaxy_handler.data_columns}")
    
    return galaxy_handler


def example_binary_accelerometer():
    """Example using binary format with accelerometer data."""
    print("\n=== Binary Accelerometer Example ===")
    
    preprocess_args = {
        'required_daily_coverage': 0.5,
        'autocalib_sphere_crit': 1,
        'autocalib_sd_criter': 0.3,
    }
    
    # Using the unified handler for binary accelerometer data
    galaxy_handler = GalaxyDataHandler(
        galaxy_file_path='../data/smartwatch/binary_data/', 
        data_format='binary', 
        data_type='accelerometer',  # Explicitly set (default would be 'accelerometer' for binary)
        # time_column defaults to 'unix_timestamp_in_ms' for binary format
        # data_columns defaults to ['acceleration_x', 'acceleration_y', 'acceleration_z'] for binary format
        verbose=True, 
        preprocess_args=preprocess_args
    )
    
    print(f"Data format: {galaxy_handler.data_format}")
    print(f"Data type: {galaxy_handler.data_type}")
    print(f"Time column: {galaxy_handler.time_column}")
    print(f"Data columns: {galaxy_handler.data_columns}")
    
    # Get metadata
    meta_data = galaxy_handler.get_meta_data()
    print(f"Meta data: {meta_data}")
    
    return galaxy_handler


def example_binary_custom_columns():
    """Example using binary format with custom column names."""
    print("\n=== Binary Accelerometer Example (Custom Columns) ===")
    
    preprocess_args = {
        'required_daily_coverage': 0.5,
    }
    
    # Using the unified handler for binary accelerometer data with custom column names
    galaxy_handler = GalaxyDataHandler(
        galaxy_file_path='../data/smartwatch/binary_data/', 
        data_format='binary', 
        data_type='accelerometer',  # Explicitly set (default would be 'accelerometer' for binary)
        time_column='custom_timestamp',  # Custom time column name
        data_columns=['accel_x', 'accel_y', 'accel_z'],  # Custom data column names
        verbose=True, 
        preprocess_args=preprocess_args
    )
    
    print(f"Data format: {galaxy_handler.data_format}")
    print(f"Data type: {galaxy_handler.data_type}")
    print(f"Time column: {galaxy_handler.time_column}")
    print(f"Data columns: {galaxy_handler.data_columns}")
    
    return galaxy_handler


def example_with_defaults():
    """Example using default parameters (binary accelerometer)."""
    print("\n=== Default Parameters Example ===")
    
    # Using default parameters (binary format, accelerometer data type)
    galaxy_handler = GalaxyDataHandler(
        galaxy_file_path='../data/smartwatch/binary_data/',
        verbose=True
    )
    
    print(f"Data format: {galaxy_handler.data_format}")
    print(f"Data type: {galaxy_handler.data_type}")
    print(f"Time column: {galaxy_handler.time_column}")
    print(f"Data columns: {galaxy_handler.data_columns}")
    
    return galaxy_handler


def example_csv_with_defaults():
    """Example using CSV format with all defaults."""
    print("\n=== CSV Default Parameters Example ===")
    
    # Using default parameters for CSV format (enmo data type)
    galaxy_handler = GalaxyDataHandler(
        galaxy_file_path='../data/smartwatch/sample1.csv',
        data_format='csv',
        verbose=True
    )
    
    print(f"Data format: {galaxy_handler.data_format}")
    print(f"Data type: {galaxy_handler.data_type}")
    print(f"Time column: {galaxy_handler.time_column}")
    print(f"Data columns: {galaxy_handler.data_columns}")
    
    return galaxy_handler


if __name__ == "__main__":
    # Note: These examples assume the data files exist
    # Uncomment the examples you want to run
    
    # example_csv_enmo()
    # example_csv_enmo_defaults()
    # example_binary_accelerometer()
    # example_binary_custom_columns()
    # example_with_defaults()
    # example_csv_with_defaults()
    
    print("Examples ready to run. Uncomment the desired example in the main block.") 