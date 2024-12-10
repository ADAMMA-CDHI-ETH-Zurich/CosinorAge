import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.features.utils.cosinor_analysis import cosinor_by_day, cosinor_multiday

@pytest.fixture
def sample_data():
    # Create 2 days of synthetic data with known rhythmic pattern
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-02 23:59:00',
        freq='1min'
    )
    
    # Generate synthetic activity data with known parameters
    time_minutes = np.arange(len(dates))
    mesor = 0.15
    amplitude = 0.1
    acrophase = np.pi/2  # Peak at 6 hours
    
    enmo = mesor + amplitude * np.cos(2*np.pi*time_minutes/1440 - acrophase)
    # Add some noise
    enmo += np.random.normal(0, 0.02, len(dates))
    
    df = pd.DataFrame({
        'ENMO': enmo
    }, index=dates)
    
    return df

def test_cosinor_by_day_basic_functionality(sample_data):
    params_df, fitted_vals = cosinor_by_day(sample_data)
    
    # Check return types
    assert isinstance(params_df, pd.DataFrame)
    assert isinstance(fitted_vals, pd.DataFrame)
    
    # Check expected columns
    expected_columns = ['mesor', 'amplitude', 'acrophase', 'acrophase_time']
    assert all(col in params_df.columns for col in expected_columns)
    
    # Check number of days
    assert len(params_df) == 2  # Should have results for 2 days

def test_cosinor_multiday_basic_functionality(sample_data):
    params, fitted_vals = cosinor_multiday(sample_data)
    
    # Check return types
    assert isinstance(params, dict)
    assert isinstance(fitted_vals, pd.Series)
    
    # Check expected keys
    expected_keys = ['mesor', 'amplitude', 'acrophase', 'acrophase_time']
    assert all(key in params for key in expected_keys)

def test_invalid_input_missing_column():
    df = pd.DataFrame({
        'Wrong_Column': [1, 2, 3]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))
    
    with pytest.raises(ValueError, match="must have a Timestamp index and an 'ENMO' column"):
        cosinor_by_day(df)
    
    with pytest.raises(ValueError, match="must have a Timestamp index and an 'ENMO' column"):
        cosinor_multiday(df)

def test_invalid_input_wrong_length():
    # Create data that's not a multiple of 1440 minutes
    df = pd.DataFrame({
        'ENMO': [1] * 1441  # One minute extra
    }, index=pd.date_range('2024-01-01', periods=1441, freq='1min'))
    
    with pytest.raises(ValueError, match="Data length is not a multiple of a day"):
        cosinor_by_day(df)
    
    with pytest.raises(ValueError, match="Data length is not a multiple of a day"):
        cosinor_multiday(df)

def test_parameter_ranges(sample_data):
    # Test by-day parameters
    params_df, _ = cosinor_by_day(sample_data)
    
    # Check parameter ranges
    assert all(0 <= params_df['acrophase_time']) and all(params_df['acrophase_time'] <= 24)
    assert all(params_df['amplitude'] >= 0)
    assert all(0 <= params_df['acrophase']) and all(params_df['acrophase'] <= 2*np.pi)
    
    # Test multiday parameters
    params, _ = cosinor_multiday(sample_data)
    
    assert 0 <= params['acrophase_time'] <= 24
    assert params['amplitude'] >= 0
    assert 0 <= params['acrophase'] <= 2*np.pi

def test_fitted_values_length(sample_data):
    _, fitted_vals_by_day = cosinor_by_day(sample_data)
    _, fitted_vals_multiday = cosinor_multiday(sample_data)
    
    # Check that fitted values match input length
    assert len(fitted_vals_by_day) == len(sample_data)
    assert len(fitted_vals_multiday) == len(sample_data)