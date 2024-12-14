import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.features.utils.cosinor_analysis import cosinor_multiday

def create_test_data(days=1, amplitude=1, mesor=0, phase_shift=0):
    """Helper function to create synthetic test data"""
    minutes = 1440 * days
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(minutes)]
    time = np.arange(minutes)
    # Create perfect cosine wave with known parameters
    enmo = mesor + amplitude * np.cos(2 * np.pi * time/1440 + phase_shift)
    
    return pd.DataFrame({
        'ENMO': enmo
    }, index=pd.DatetimeIndex(timestamps))

def test_cosinor_multiday_basic_functionality():
    """Test basic functionality with perfect cosine data"""
    # Create test data with known parameters
    test_df = create_test_data(days=1, amplitude=1, mesor=2, phase_shift=0)
    
    # Run cosinor analysis
    params, fitted_vals = cosinor_multiday(test_df)
    
    # Check if parameters are close to expected values
    assert np.isclose(params['mesor'], 2, atol=0.1)
    assert np.isclose(params['amplitude'], 1, atol=0.1)
    assert isinstance(fitted_vals, pd.Series)
    assert len(fitted_vals) == 1440

def test_cosinor_multiday_multiple_days():
    """Test with multiple days of data"""
    test_df = create_test_data(days=3, amplitude=1, mesor=2)
    
    params, fitted_vals = cosinor_multiday(test_df)
    
    assert np.isclose(params['mesor'], 2, atol=0.1)
    assert np.isclose(params['amplitude'], 1, atol=0.1)
    assert len(fitted_vals) == 4320  # 3 days * 1440 minutes

def test_cosinor_multiday_phase_shift():
    """Test with phase-shifted data"""
    phase_shift = np.pi/2  # 6-hour shift
    test_df = create_test_data(days=1, amplitude=1, mesor=0, phase_shift=phase_shift)
    
    params, _ = cosinor_multiday(test_df)
    
    expected_acrophase_time = (24 - 6) * 60  # Should be 18:00 (1080 minutes)
    assert np.isclose(params['acrophase_time'], expected_acrophase_time, atol=30)  # Allow 30 minutes tolerance

def test_invalid_input_no_enmo():
    """Test error handling for missing ENMO column"""
    df = pd.DataFrame({
        'wrong_column': [1, 2, 3]
    }, index=pd.date_range('2024-01-01', periods=3, freq='min'))  # Changed 'T' to 'min'
    
    with pytest.raises(ValueError, match="must have.*ENMO.*column"):
        cosinor_multiday(df)

def test_invalid_input_no_datetime_index():
    """Test error handling for missing datetime index"""
    df = pd.DataFrame({
        'ENMO': [1, 2, 3]
    })
    
    with pytest.raises(ValueError, match="must have a Timestamp index"):
        cosinor_multiday(df)

def test_invalid_input_wrong_length():
    """Test error handling for data not multiple of 1440"""
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(1000)]
    df = pd.DataFrame({
        'ENMO': np.random.random(1000)
    }, index=pd.DatetimeIndex(timestamps))
    
    with pytest.raises(ValueError, match="Data length is not a multiple of a day"):
        cosinor_multiday(df)

def test_cosinor_multiday_output_types():
    """Test output types and structure"""
    test_df = create_test_data(days=1)
    
    params, fitted_vals = cosinor_multiday(test_df)
    
    # Check parameter dictionary structure
    assert isinstance(params, dict)
    assert all(key in params for key in ['mesor', 'amplitude', 'acrophase', 'acrophase_time'])
    assert all(isinstance(val, float) for val in params.values())
    
    # Check fitted values
    assert isinstance(fitted_vals, pd.Series)
    assert len(fitted_vals) == len(test_df)

def test_cosinor_multiday_noise_robustness():
    """Test function's robustness to noisy data"""
    # Create data with noise
    test_df = create_test_data(days=1, amplitude=1, mesor=2)
    noise = np.random.normal(0, 0.1, size=1440)
    test_df['ENMO'] += noise
    
    params, _ = cosinor_multiday(test_df)
    
    # Check if parameters are still reasonably close to expected values
    assert np.isclose(params['mesor'], 2, atol=0.2)
    assert np.isclose(params['amplitude'], 1, atol=0.2)