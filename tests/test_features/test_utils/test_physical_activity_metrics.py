import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.features.utils.physical_activity_metrics import activity_metrics

def test_activity_metrics_empty_data():
    """Test behavior with empty input data"""
    empty_df = pd.DataFrame(columns=['ENMO'])
    result = activity_metrics(empty_df)
    assert result == ([], [], [], [])

def test_activity_metrics_single_day():
    """Test calculation for a single day with known values"""
    dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    values = np.zeros(1440)
    # Using the default cutpoints:
    # sl = 0.030, lm = 0.100, mv = 0.400
    values[:360] = 0.05  # Light activity (0.03 < x ≤ 0.1)
    values[360:720] = 0.2  # Moderate activity (0.1 < x ≤ 0.4)
    values[720:1080] = 0.5  # Vigorous activity (x > 0.4)
    values[1080:] = 0.02  # Sedentary (x ≤ 0.03)
    
    data = pd.DataFrame({'ENMO': values}, index=dates)
    
    sed, light, mod, vig = activity_metrics(data)
    
    assert sed == [360]  # Last 6 hours (≤ 0.03)
    assert light == [360]  # First 6 hours (0.03-0.1)
    assert mod == [360]  # Second 6 hours (0.1-0.4)
    assert vig == [360]  # Third 6 hours (> 0.4)

def test_activity_metrics_multiple_days():
    """Test calculation for multiple days"""
    dates = pd.date_range('2023-01-01', periods=4320, freq='min')  # 3 days
    values = np.ones(4320) * 0.05  # All light activity
    data = pd.DataFrame({'ENMO': values}, index=dates)
    
    sed, light, mod, vig = activity_metrics(data)
    
    assert len(sed) == 3
    assert len(light) == 3
    assert len(mod) == 3
    assert len(vig) == 3
    assert all(x == 0 for x in sed)  # No sedentary time
    assert all(x == 1440 for x in light)  # All light activity
    assert all(x == 0 for x in mod)  # No moderate activity
    assert all(x == 0 for x in vig)  # No vigorous activity

def test_activity_metrics_custom_cutpoints():
    """Test with custom cutpoint values"""
    dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    values = np.ones(1440) * 0.2
    data = pd.DataFrame({'ENMO': values}, index=dates)
    
    custom_cutpoints = {
        "sl": 0.1,
        "lm": 0.3,
        "mv": 0.5
    }
    
    sed, light, mod, vig = activity_metrics(data, custom_cutpoints)
    
    # With ENMO = 0.2:
    # 0.1 < 0.2 ≤ 0.3 -> Moderate activity
    assert sed == [0]  # No sedentary (> 0.1)
    assert light == [0]  # No light (0.1-0.3)
    assert mod == [1440]  # All moderate (between lm and mv)
    assert vig == [0]  # No vigorous (> 0.5)

def test_activity_metrics_cutpoints_behavior():
    """Test behavior with different cutpoint configurations"""
    dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    values = np.ones(1440) * 0.2
    data = pd.DataFrame({'ENMO': values}, index=dates)
    
    # Test with empty cutpoints - should use defaults
    empty_cutpoints = {}
    sed, light, mod, vig = activity_metrics(data, empty_cutpoints)
    
    # With default cutpoints and ENMO = 0.2:
    # sl = 0.030, lm = 0.100, mv = 0.400
    # 0.2 is between lm (0.1) and mv (0.4), so it should be moderate activity
    assert sed == [0]      # Not ≤ 0.03
    assert light == [0]    # Not between 0.03 and 0.1
    assert mod == [1440]   # Between 0.1 and 0.4
    assert vig == [0]      # Not > 0.4
    
    # Test with custom cutpoints
    custom_cutpoints = {
        "pa_cutpoint_sl": 0.25,  # Higher than the value (0.2)
        "pa_cutpoint_lm": 0.3,
        "pa_cutpoint_mv": 0.5
    }
    sed2, light2, mod2, vig2 = activity_metrics(data, custom_cutpoints)
    
    # With ENMO = 0.2 and cutpoints:
    # sl = 0.25, lm = 0.3, mv = 0.5
    # 0.2 is less than sl (0.25)
    assert sed2 == [1440]  # All minutes are sedentary (≤ 0.25)
    assert light2 == [0]   # None between 0.25 and 0.3
    assert mod2 == [0]     # None between 0.3 and 0.5
    assert vig2 == [0]     # None > 0.5
    
    # Test with alternative key format
    alt_cutpoints = {
        "pa_cutpoint_sl": 0.15,  # Less than the value (0.2)
        "pa_cutpoint_lm": 0.3,   # Greater than the value (0.2)
        "pa_cutpoint_mv": 0.5
    }
    sed3, light3, mod3, vig3 = activity_metrics(data, alt_cutpoints)
    
    # With ENMO = 0.2 and cutpoints:
    # sl = 0.15, lm = 0.3, mv = 0.5
    # 0.2 is between sl (0.15) and lm (0.3)
    assert sed3 == [0]      # Not ≤ 0.15
    assert light3 == [1440] # Between 0.15 and 0.3
    assert mod3 == [0]      # Not between 0.3 and 0.5
    assert vig3 == [0]      # Not > 0.5

def test_activity_metrics_invalid_data_format():
    """Test behavior with invalid data format"""
    dates = pd.date_range('2023-01-01', periods=1440, freq='min')
    values = np.ones(1440) * 0.2
    data = pd.DataFrame({'wrong_name': values}, index=dates)
    
    with pytest.raises(KeyError):
        activity_metrics(data)