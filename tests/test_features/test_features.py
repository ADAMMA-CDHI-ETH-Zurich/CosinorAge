import pytest
import pandas as pd
import numpy as np
from cosinorage.features.features import WearableFeatures
from cosinorage.datahandlers import DataHandler


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create exactly 7 days of minute-level data (7 * 1440 minutes)
    dates = pd.date_range(start='2024-01-01', 
                         periods=7*1440,  # Exactly 7 days worth of minutes
                         freq='1min')
    
    # Create synthetic ENMO data with a daily pattern
    enmo = np.sin(2 * np.pi * np.arange(len(dates)) / 1440) + 1
    
    # Create sleep data (0 for sleep, 1 for wake)
    # Simple pattern: sleep from 23:00 to 07:00
    sleep = np.ones(len(dates))  # Initialize all as wake
    for i in range(len(dates)):
        hour = dates[i].hour
        if hour >= 23 or hour < 7:
            sleep[i] = 0  # Set as sleep during night hours
    
    # Create DataFrame with all required columns
    df = pd.DataFrame({
        'ENMO': enmo,
        'sleep': sleep,
        'date': dates.date,
        'time': range(len(dates)),
        'cos': np.cos(2 * np.pi * np.arange(len(dates)) / 1440),
        'sin': np.sin(2 * np.pi * np.arange(len(dates)) / 1440)
    }, index=dates)
    
    # Add cosinor_fitted column
    df['cosinor_fitted'] = df['ENMO'].rolling(window=1440, center=True).mean()
    
    return df

@pytest.fixture
def mock_data_handler(sample_data):
    """Create a mock DataHandler"""
    class MockDataHandler:
        def get_ml_data(self):
            return sample_data
    return MockDataHandler()

def test_wearable_features_initialization(mock_data_handler):
    """Test basic initialization of WearableFeatures"""
    features = WearableFeatures(mock_data_handler)
    assert isinstance(features.ml_data, pd.DataFrame)
    assert isinstance(features.feature_dict, dict)

def test_cosinor_features(mock_data_handler):
    """Test cosinor feature computation"""
    features = WearableFeatures(mock_data_handler)
    cosinor_dict = features.feature_dict['cosinor']
    
    # Check if all expected features are present
    assert 'mesor' in cosinor_dict
    assert 'amplitude' in cosinor_dict
    assert 'acrophase' in cosinor_dict
    assert 'acrophase_time' in cosinor_dict
    
    # Check value ranges
    assert isinstance(cosinor_dict['mesor'], (float, np.float64))
    assert isinstance(cosinor_dict['amplitude'], (float, np.float64))
    assert -2*np.pi <= cosinor_dict['acrophase'] <= 2*np.pi
    assert 0 <= cosinor_dict['acrophase_time'] <= 1440

def test_nonparam_features(mock_data_handler):
    """Test non-parametric feature computation"""
    features = WearableFeatures(mock_data_handler)
    nonparam_dict = features.feature_dict['nonparam']
    
    # Check if all expected features are present
    assert 'IS' in nonparam_dict
    assert 'IV' in nonparam_dict
    assert 'M10' in nonparam_dict
    assert 'L5' in nonparam_dict
    assert 'RA' in nonparam_dict
    
    # Check value ranges based on type
    if isinstance(nonparam_dict['IS'], (list, np.ndarray)):
        assert all(0 <= x <= 1 for x in nonparam_dict['IS'])
    else:
        assert 0 <= nonparam_dict['IS'] <= 1

    if isinstance(nonparam_dict['IV'], (list, np.ndarray)):
        assert all(x >= 0 for x in nonparam_dict['IV'])
    else:
        assert nonparam_dict['IV'] >= 0

    # M10 and L5 are expected to be lists
    assert isinstance(nonparam_dict['M10'], (list, np.ndarray))
    assert isinstance(nonparam_dict['L5'], (list, np.ndarray))
    assert all(x > y for x, y in zip(nonparam_dict['M10'], nonparam_dict['L5']))

    if isinstance(nonparam_dict['RA'], (list, np.ndarray)):
        assert all(0 <= x <= 1 for x in nonparam_dict['RA'])
    else:
        assert 0 <= nonparam_dict['RA'] <= 1

def test_physical_activity_metrics(mock_data_handler):
    """Test physical activity metrics computation"""
    features = WearableFeatures(mock_data_handler)
    pa_dict = features.feature_dict['physical_activity']
    
    # Check if all expected features are present
    assert 'sedentary' in pa_dict
    assert 'light' in pa_dict
    assert 'moderate' in pa_dict
    assert 'vigorous' in pa_dict
    
    # Check value ranges for each metric
    for key in pa_dict:
        if isinstance(pa_dict[key], (list, np.ndarray)):
            assert all(x >= 0 for x in pa_dict[key])  # Values should be non-negative
            # Print the actual range for debugging
            print(f"{key} range: {min(pa_dict[key])} to {max(pa_dict[key])}")
        else:
            assert pa_dict[key] >= 0  # Values should be non-negative
            print(f"{key} value: {pa_dict[key]}")

def test_sleep_metrics(mock_data_handler):
    """Test sleep metrics computation"""
    features = WearableFeatures(mock_data_handler)
    sleep_dict = features.feature_dict['sleep']
    
    # Check if all expected features are present
    assert 'TST' in sleep_dict
    assert 'WASO' in sleep_dict
    assert 'PTA' in sleep_dict
    assert 'NWB' in sleep_dict
    assert 'SOL' in sleep_dict
    assert 'SRI' in sleep_dict
    
    # Check value ranges for each metric
    metrics_ranges = {
        'TST': (0, float('inf')),  # Total Sleep Time in minutes
        'WASO': (0, float('inf')),  # Wake After Sleep Onset in minutes
        'PTA': (0, 100),  # Percentage Time Asleep
        'NWB': (0, float('inf')),  # Number of Wake Bouts
        'SOL': (0, float('inf')),  # Sleep Onset Latency in minutes
        'SRI': (0, 100)  # Sleep Regularity Index (adjusted to percentage)
    }
    
    for key, (min_val, max_val) in metrics_ranges.items():
        if isinstance(sleep_dict[key], (list, np.ndarray)):
            assert all(min_val <= x <= max_val for x in sleep_dict[key])
            # Print the actual range for debugging
            print(f"{key} range: {min(sleep_dict[key])} to {max(sleep_dict[key])}")
        else:
            assert min_val <= sleep_dict[key] <= max_val
            print(f"{key} value: {sleep_dict[key]}")

def test_get_features(mock_data_handler):
    """Test get_features method"""
    features = WearableFeatures(mock_data_handler)
    feature_dict = features.get_features()
    
    assert isinstance(feature_dict, dict)
    assert 'cosinor' in feature_dict
    assert 'nonparam' in feature_dict
    assert 'physical_activity' in feature_dict
    assert 'sleep' in feature_dict

def test_get_ml_data(mock_data_handler):
    """Test get_ml_data method"""
    features = WearableFeatures(mock_data_handler)
    ml_data = features.get_ml_data()
    
    assert isinstance(ml_data, pd.DataFrame)
    assert 'ENMO' in ml_data.columns
    assert isinstance(ml_data.index, pd.DatetimeIndex)
    assert len(ml_data) % 1440 == 0  # Check if data length is multiple of 1440

def test_invalid_data_handling():
    """Test handling of invalid data"""
    # Create data with NaN values
    dates = pd.date_range(start='2024-01-01', end='2024-01-08', freq='1min')
    enmo = np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 60)) + 1
    invalid_data = pd.DataFrame({'ENMO': enmo}, index=dates)
    invalid_data.loc[invalid_data.index[0:100], 'ENMO'] = np.nan
    
    class MockInvalidDataHandler:
        def get_ml_data(self):
            return invalid_data
    
    with pytest.raises(Exception):
        features = WearableFeatures(MockInvalidDataHandler())