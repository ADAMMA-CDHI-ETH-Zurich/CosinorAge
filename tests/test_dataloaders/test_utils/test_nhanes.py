import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.datahandlers.utils.nhanes import (
    read_nhanes_data,
    filter_nhanes_data,
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

def test_filter_nhanes_data(sample_nhanes_df):
    meta_dict = {'raw_data_frequency': 1/60}  # 1 sample per minute
    
    # Run function
    result = filter_nhanes_data(sample_nhanes_df, meta_dict)
    
    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert 'n_days' in meta_dict
    assert meta_dict['n_days'] == 5  # Now we expect 5 days
    assert len(np.unique(result.index.date)) == 5

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
