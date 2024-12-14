import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cosinorage.features.utils.nonparam_analysis import IS, IV, M10, L5, RA

@pytest.fixture
def sample_data():
    # Create sample data for 48 hours (2 days) with 1-minute resolution
    dates = pd.date_range(
        start='2024-01-01', 
        end='2024-01-02 23:59:00', 
        freq='1min'
    )
    # Create synthetic activity data with known pattern
    # Higher values during day (8:00-18:00), lower at night
    activity = []
    for dt in dates:
        if 8 <= dt.hour < 18:
            activity.append(10 + np.random.normal(0, 1))  # Active hours
        else:
            activity.append(1 + np.random.normal(0, 0.1))  # Rest hours

    return pd.DataFrame({'ENMO': activity}, index=dates)

def test_IS_empty_data():
    empty_data = pd.DataFrame({'ENMO': []})
    assert np.isnan(IS(empty_data))

def test_IS_normal_data(sample_data):
    is_value = IS(sample_data)
    assert isinstance(is_value, float)
    assert 0 <= is_value <= 1  # IS should be between 0 and 1

def test_IV_empty_data():
    empty_data = pd.DataFrame({'ENMO': []})
    assert np.isnan(IV(empty_data))

def test_IV_normal_data(sample_data):
    iv_value = IV(sample_data)
    assert isinstance(iv_value, float)
    assert iv_value >= 0  # IV should be non-negative

def test_M10_empty_data():
    empty_data = pd.DataFrame({'ENMO': []})
    m10_values, m10_starts = M10(empty_data)
    assert len(m10_values) == 0
    assert len(m10_starts) == 0

def test_M10_normal_data(sample_data):
    m10_values, m10_starts = M10(sample_data)
    assert len(m10_values) == 2  # Should have values for 2 days
    assert all(isinstance(x, float) for x in m10_values)
    assert all(isinstance(x, pd.Timestamp) for x in m10_starts)

def test_L5_empty_data():
    empty_data = pd.DataFrame({'ENMO': []})
    l5_values, l5_starts = L5(empty_data)
    assert len(l5_values) == 0
    assert len(l5_starts) == 0

def test_L5_normal_data(sample_data):
    l5_values, l5_starts = L5(sample_data)
    assert len(l5_values) == 2  # Should have values for 2 days
    assert all(isinstance(x, float) for x in l5_values)
    assert all(isinstance(x, pd.Timestamp) for x in l5_starts)

def test_RA_empty_data():
    assert len(RA([], [])) == 0

def test_RA_normal_data():
    m10_values = [10, 12]
    l5_values = [2, 3]
    ra_values = RA(m10_values, l5_values)
    assert len(ra_values) == 2
    assert all(isinstance(x, float) for x in ra_values)
    assert all(0 <= x <= 1 for x in ra_values)  # RA should be between 0 and 1

def test_RA_mismatched_lengths():
    with pytest.raises(ValueError):
        RA([1, 2], [1])

def test_constant_data():
    # Create constant data to test edge cases
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
    constant_data = pd.DataFrame({'ENMO': [1] * len(dates)}, index=dates)
    
    # Test IS with constant data
    assert np.isnan(IS(constant_data))  # Should return nan for constant data
    
    # Test IV with constant data
    assert np.isnan(IV(constant_data))  # Should return nan for constant data

def test_data_validation():
    # Test with invalid data types
    with pytest.raises(TypeError):
        IS([1, 2, 3])  # List instead of DataFrame
    
    with pytest.raises(TypeError):
        IV([1, 2, 3])  # List instead of DataFrame

    # Test with DataFrame but missing required column
    invalid_df = pd.DataFrame({'Wrong_Column': [1, 2, 3]})
    with pytest.raises(KeyError):
        IS(invalid_df)
    
    with pytest.raises(KeyError):
        IV(invalid_df)

    # Test with DataFrame but no datetime index
    invalid_df = pd.DataFrame({'ENMO': [1, 2, 3]})
    with pytest.raises(TypeError, match="Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex"):
        IS(invalid_df)
    
    with pytest.raises(TypeError, match="Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex"):
        IV(invalid_df)

    # Test with proper datetime index but invalid data
    dates = pd.date_range(start='2024-01-01', periods=3, freq='1h')  # Changed from '1H' to '1h'
    invalid_df = pd.DataFrame({'ENMO': ['a', 'b', 'c']}, index=dates)
    with pytest.raises(TypeError):
        IS(invalid_df)
    
    with pytest.raises(TypeError):
        IV(invalid_df)