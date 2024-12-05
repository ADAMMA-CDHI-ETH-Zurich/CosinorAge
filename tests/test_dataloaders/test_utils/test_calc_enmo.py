import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from cosinorage.datahandlers.utils.calc_enmo import calculate_enmo, calculate_minute_level_enmo

def test_calculate_enmo_normal_case():
    # Test with realistic accelerometer data at 1Hz frequency
    timestamps = pd.date_range(
        start='2024-01-01 00:00:00',
        periods=180,
        freq='1s'
    )
    
    # Generate repeating pattern of the original test values
    x_values = np.tile([-0.2, 0.8, -0.5, 1.2, 0.0], 36)  # 36 * 5 = 180
    y_values = np.tile([0.9, -0.3, 1.1, 0.4, 0.7], 36)
    z_values = np.tile([0.8, 1.1, 0.6, -0.3, 1.2], 36)
    
    acc_data = pd.DataFrame({
        'TIMESTAMP': timestamps,
        'X': x_values,
        'Y': y_values,
        'Z': z_values
    }).set_index('TIMESTAMP')
    
    result = calculate_enmo(acc_data)
    
    # Expected values also repeated 36 times
    expected = np.tile([0.221, 0.393, 0.349, 0.300, 0.389], 36)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)

def test_calculate_enmo_all_zeros():
    # Test with all zeros (should return all zeros as ENMO)
    acc_data = pd.DataFrame({
        'X': [0.0, 0.0, 0.0],
        'Y': [0.0, 0.0, 0.0],
        'Z': [0.0, 0.0, 0.0]
    })
    result = calculate_enmo(acc_data)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result, expected)

def test_calculate_enmo_missing_columns():
    # Test error handling when columns are missing
    acc_data = pd.DataFrame({
        'X': [0.5, 1.0],
        'Wrong_Column': [0.5, 0.0]
    })
    result = calculate_enmo(acc_data)
    assert np.isnan(result)

def test_calculate_minute_level_enmo_normal_case():
    # Test with realistic accelerometer data at 1Hz frequency
    timestamps = pd.date_range(
        start='2024-01-01 00:00:00',
        periods=180,
        freq='1s'
    )
    
    # Generate repeating pattern of the original test values
    x_values = np.tile([-0.2, 0.8, -0.5, 1.2, 0.0], 36)  # 36 * 5 = 180
    y_values = np.tile([0.9, -0.3, 1.1, 0.4, 0.7], 36)
    z_values = np.tile([0.8, 1.1, 0.6, -0.3, 1.2], 36)
    
    acc_data = pd.DataFrame({
        'TIMESTAMP': timestamps,
        'X': x_values,
        'Y': y_values,
        'Z': z_values
    }).set_index('TIMESTAMP')
    
    result = calculate_enmo(acc_data)
    
    # Expected values also repeated 36 times
    expected = np.tile([0.221, 0.393, 0.349, 0.300, 0.389], 36)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)

def test_calculate_minute_level_enmo_empty():
    # Test with empty DataFrame
    empty_df = pd.DataFrame({
        'TIMESTAMP': pd.DatetimeIndex([]),
        'ENMO': []
    }).set_index('TIMESTAMP')
    
    result = calculate_minute_level_enmo(empty_df, 1)
    assert result.empty