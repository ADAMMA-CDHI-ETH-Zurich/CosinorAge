import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from unittest.mock import patch, mock_open, MagicMock

from cosinorage.datahandlers.utils.nhanes import (
    read_nhanes_data,
    filter_and_preprocess_nhanes_data,
    resample_nhanes_data,
    remove_bytes,
    clean_data,
    calculate_measure_time
)

@pytest.fixture
def sample_nhanes_df():
    # Create 5 consecutive days of data at 1-minute intervals
    index = pd.date_range(
        start='2023-01-01 00:00:00',
        end='2023-01-05 23:59:00',  # End at the last minute of day 5
        freq='min'
    )
    
    data = {
        'X': np.random.normal(0, 1, len(index)),
        'Y': np.random.normal(0, 1, len(index)),
        'Z': np.random.normal(0, 1, len(index)),
        'wear': np.ones(len(index)),
        'sleep': np.zeros(len(index)),
        'paxpredm': np.ones(len(index)),
        'ENMO': np.random.uniform(0, 1, len(index))
    }
    return pd.DataFrame(data, index=index)

@pytest.fixture
def sample_bytes_df():
    return pd.DataFrame({
        'text_col': [b'hello', b'world'],
        'num_col': [1, 2],
        'mixed_col': [b'test', 'normal']
    })

@pytest.fixture
def mock_sas_data():
    # Mock data that would come from SAS files
    day_data = pd.DataFrame({
        'SEQN': ['1234'],  # Keep uppercase for initial check
        'paxqfd': [0],     # Quality flag < 1
        'paxwwmd': [1020], # 17 hours of wear time
        'paxswmd': [120],  # 2 hours of sleep
        'valid_hours': [19] # More than 16 valid hours
    })
    
    # Create minute data for 4 complete days (to pass the minimum days requirement)
    n_minutes = 4 * 24 * 60  # 4 days of minute-level data
    
    # Create all arrays first to ensure they have the same length
    random_x = np.random.normal(0, 1, n_minutes).tolist()
    random_y = np.random.normal(0, 1, n_minutes).tolist()
    random_z = np.random.normal(0, 1, n_minutes).tolist()
    day_numbers = np.repeat(range(1, 5), 24 * 60).tolist()  # 4 days numbered 1-4
    ssnmp_values = list(range(0, n_minutes * 80, 80))  # 80 measurements per second
    
    # Create hour and minute values for each measurement
    hours = np.tile(range(24), 4 * 60)  # 24 hours repeated for 4 days
    minutes = np.repeat(range(60), 4 * 24)  # 60 minutes repeated for each hour
    
    # Calculate epochs (12 per hour, 5 minutes per epoch)
    epochs = (12 * hours + np.floor(minutes / 5) + 1).astype(int)
    
    # Verify all arrays have the same length
    assert len(random_x) == n_minutes
    assert len(random_y) == n_minutes
    assert len(random_z) == n_minutes
    assert len(day_numbers) == n_minutes
    assert len(ssnmp_values) == n_minutes
    assert len(epochs) == n_minutes
    assert len(hours) == n_minutes
    assert len(minutes) == n_minutes
    
    min_data = pd.DataFrame({
        'SEQN': ['1234'] * n_minutes,      # Keep uppercase for initial check
        'PAXMTSM': [0.5] * n_minutes,      # Valid measurement value (!= -0.01)
        'PAXPREDM': [1] * n_minutes,       # Valid wear/non-sleep (not in [3, 4])
        'PAXQFM': [0] * n_minutes,         # Good quality (< 1)
        'paxmxm': random_x,
        'paxmym': random_y,
        'paxmzm': random_z,
        'paxdaym': day_numbers,
        'paxssnmp': ssnmp_values
    })
    
    head_data = pd.DataFrame({
        'SEQN': ['1234'],                  # Keep uppercase for initial check
        'paxftime': ['00:00:00'],          # Start at midnight for simplicity
        'paxfday': ['Monday']
    })
    
    # Let the function handle column renaming
    
    return day_data, min_data, head_data

def test_remove_bytes():
    # Test data
    df = pd.DataFrame({
        'text_col': [b'hello', b'world'],
        'num_col': [1, 2]
    })
    
    # Run function
    result = remove_bytes(df)
    
    # Assertions
    assert isinstance(result['text_col'][0], str)
    assert result['text_col'][0] == 'hello'
    assert result['num_col'][0] == 1

def test_clean_data():
    # Test data
    days_df = pd.DataFrame({'seqn': [1, 2]})
    data_df = pd.DataFrame({
        'SEQN': [1, 1, 2, 3],
        'PAXMTSM': [0.5, -0.01, 0.3, 0.4],
        'PAXPREDM': [1, 2, 3, 1],
        'PAXQFM': [0, 0.5, 0.8, 1.1]
    })
    
    # Run function
    result = clean_data(data_df, days_df)
    
    # Assertions
    assert len(result) == 1  # Only one row should meet all criteria
    assert result['SEQN'].iloc[0] == 1

def test_calculate_measure_time():
    # Test data
    row = {
        'day1_start_time': '08:00:00',
        'paxssnmp': 80  # 1 second worth of measurements
    }
    
    # Run function
    result = calculate_measure_time(row)
    
    # Assertions
    expected = datetime.strptime('08:00:01', '%H:%M:%S')
    assert result == expected

def test_filter_and_preprocess_nhanes_data(sample_nhanes_df):
    meta_dict = {}
    result = filter_and_preprocess_nhanes_data(sample_nhanes_df, meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    assert 'ENMO' in result.columns
    assert 'n_days' in meta_dict
    assert all(col in result.columns for col in ['X_raw', 'Y_raw', 'Z_raw'])
    # Check MIMS to mg conversion
    assert np.allclose(result['X'], sample_nhanes_df['X'] / 9.81, rtol=1e-10)

def test_resample_nhanes_data(sample_nhanes_df):
    # Create gaps in the data
    sample_nhanes_df = sample_nhanes_df.iloc[::2]  # Take every other row
    
    # Run function
    result = resample_nhanes_data(sample_nhanes_df)
    
    # Assertions
    expected_minutes = 5 * 24 * 60  # 5 days * 24 hours * 60 minutes
    assert len(result) == expected_minutes - 1  # Account for end time at 23:59
    assert result.index.freq == pd.Timedelta('1 min')
    # Check that there are no gaps in the resampled data
    assert (result.index[1] - result.index[0]) == pd.Timedelta('1 min')
    # Check start and end times
    assert result.index[0].strftime('%Y-%m-%d %H:%M:%S') == '2023-01-01 00:00:00'
    assert result.index[-1].strftime('%Y-%m-%d %H:%M:%S') == '2023-01-05 23:58:00'

def test_read_nhanes_data_missing_seqn():
    with pytest.raises(ValueError, match="The seqn is required for nhanes data"):
        read_nhanes_data("dummy_dir")

def test_resample_nhanes_data_with_gaps():
    # Create sample data with gaps
    index = pd.date_range('2023-01-01', periods=100, freq='2min')
    data = pd.DataFrame({
        'X': np.random.normal(0, 1, len(index)),
        'Y': np.random.normal(0, 1, len(index)),
        'Z': np.random.normal(0, 1, len(index)),
        'wear': np.ones(len(index)),
        'sleep': np.zeros(len(index)),
        'paxpredm': np.ones(len(index))
    }, index=index)
    
    result = resample_nhanes_data(data, verbose=True)
    
    assert len(result) > len(data)  # Should have filled gaps
    assert result.index.freq == pd.Timedelta('1 min')
    assert all(result['wear'].isin([0, 1]))  # Check wear values are binary
    assert all(result['sleep'].isin([0, 1]))  # Check sleep values are binary

def test_clean_data_edge_cases():
    days_df = pd.DataFrame({'seqn': [1, 2]})
    data_df = pd.DataFrame({
        'SEQN': [1, 1, 2, 3],
        'PAXMTSM': [-0.01, 0.5, -0.01, 0.4],  # Test boundary case
        'PAXPREDM': [3, 1, 2, 4],  # Test invalid predictions
        'PAXQFM': [1, 0, 0.5, 2]  # Test quality flags
    })
    
    result = clean_data(data_df, days_df)
    
    assert len(result) == 1  # Only one row should meet all criteria
    assert result['PAXMTSM'].iloc[0] == 0.5  # Should keep valid measurement
    assert result['PAXPREDM'].iloc[0] == 1  # Should keep valid prediction

def test_calculate_measure_time_edge_cases():
    # Test midnight crossing
    row1 = {'day1_start_time': '23:59:00', 'paxssnmp': 80}
    result1 = calculate_measure_time(row1)
    assert result1.strftime('%H:%M:%S') == '23:59:01'
    
    # Test start of day
    row2 = {'day1_start_time': '00:00:00', 'paxssnmp': 0}
    result2 = calculate_measure_time(row2)
    assert result2.strftime('%H:%M:%S') == '00:00:00'

def test_remove_bytes_mixed_types():
    df = pd.DataFrame({
        'bytes_col': [b'test', b'data'],
        'str_col': ['normal', 'string'],
        'num_col': [1, 2],
        'mixed_col': [b'bytes', 'string']
    })
    
    result = remove_bytes(df)
    
    assert all(isinstance(x, str) for x in result['bytes_col'])
    assert all(isinstance(x, str) for x in result['str_col'])
    assert all(isinstance(x, (int, float)) for x in result['num_col'])
    assert all(isinstance(x, str) for x in result['mixed_col'])
