import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.features.utils.sleep_metrics import (
    apply_sleep_wake_predictions,
    waso,
    tst,
    pta,
    sri
)

@pytest.fixture
def sample_sleep_data():
    # Create exactly 48 hours of data starting at midnight
    dates = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-01-02 23:59:00',  # End at last minute of second day
        freq='1min'
    )
    
    # Create alternating sleep patterns
    sleep_states = np.zeros(len(dates))
    # Set wake periods - 200 minutes each day
    sleep_states[300:500] = 1  # Wake period in first day (200 minutes)
    sleep_states[1740:1940] = 1  # Wake period in second day (200 minutes)
    
    df = pd.DataFrame({
        'sleep': sleep_states
    }, index=dates)
    
    return df

@pytest.fixture
def sample_enmo_data():
    # Create sample ENMO data
    dates = pd.date_range(
        start='2024-01-01 12:00:00',
        end='2024-01-02 12:00:00',
        freq='1min'
    )
    
    df = pd.DataFrame({
        'ENMO': np.random.normal(0.1, 0.05, len(dates))
    }, index=dates)
    
    return df

def test_apply_sleep_wake_predictions(sample_enmo_data):
    result = apply_sleep_wake_predictions(sample_enmo_data)
    assert isinstance(result, pd.Series)
    assert set(result.unique()).issubset({0, 1})  # Should only contain 0s and 1s
    assert len(result) == len(sample_enmo_data)

def test_apply_sleep_wake_predictions_missing_enmo():
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    with pytest.raises(ValueError, match="Column ENMO not found"):
        apply_sleep_wake_predictions(df)

def test_waso(sample_sleep_data):
    result = waso(sample_sleep_data)
    assert isinstance(result, pd.Series)
    assert len(result) == 2  # Should have 2 days of data
    assert all(result >= 0)  # WASO should always be non-negative
    # Test specific values based on our sample data
    assert result.iloc[0] == 200  # First day has 200 minutes of wake after sleep onset
    assert result.iloc[1] == 200  # Second day has 200 minutes of wake after sleep onset

def test_tst(sample_sleep_data):
    result = tst(sample_sleep_data)
    assert isinstance(result, pd.Series)
    assert len(result) == 2  # Should have 2 days of data
    assert all(result >= 0)  # TST should always be non-negative
    assert all(result <= 1440)  # TST should not exceed minutes in a day

def test_pta(sample_sleep_data):
    result = pta(sample_sleep_data)
    assert isinstance(result, pd.Series)
    assert len(result) == 2  # Should have 2 days of data
    assert all(result >= 0)  # PTA should be between 0 and 1
    assert all(result <= 1)

def test_sri(sample_sleep_data):
    result = sri(sample_sleep_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # Should have 1 day of SRI (needs 2 days to calculate first value)
    assert all(result['SRI'] >= -100)  # SRI should be between -100 and 100
    assert all(result['SRI'] <= 100)
    # For our sample data, we expect high regularity since the patterns are similar
    assert result['SRI'].iloc[0] > 0  # Should be positive due to similar patterns

def test_sri_insufficient_data():
    # Create data for less than 2 days
    dates = pd.date_range(
        start='2024-01-01 12:00:00',
        end='2024-01-02 11:59:00',
        freq='1min'
    )
    df = pd.DataFrame({'sleep': np.zeros(len(dates))}, index=dates)
    
    with pytest.raises(ValueError, match="Insufficient data"):
        sri(df)