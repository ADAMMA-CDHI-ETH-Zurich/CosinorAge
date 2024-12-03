import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.features.utils.nonparam_analysis import IV, IS, RA, M10, L5

@pytest.fixture
def sample_data():
    # Create 48 hours of sample data at 1-minute intervals
    dates = pd.date_range(
        start='2024-01-01', 
        end='2024-01-02 23:59:00', 
        freq='1min'
    )
    
    # Create synthetic activity data with known patterns
    activity = np.zeros(len(dates))
    
    # Add higher activity during daytime (8AM-6PM)
    for i, dt in enumerate(dates):
        if 8 <= dt.hour < 18:  # daytime
            activity[i] = 10 + np.random.normal(0, 1)
        else:  # nighttime
            activity[i] = 2 + np.random.normal(0, 0.5)
    
    df = pd.DataFrame({
        'ENMO': activity
    }, index=dates)
    
    return df

def test_IV(sample_data):
    result = IV(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two days of data
    assert 'IV' in result.columns
    assert all(result['IV'] >= 0)  # IV should be non-negative

def test_IS(sample_data):
    result = IS(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two days of data
    assert 'IS' in result.columns
    assert all(result['IS'] >= 0)  # IS should be non-negative
    assert all(result['IS'] <= 1)  # IS should be <= 1

def test_RA(sample_data):
    result = RA(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two days of data
    assert 'RA' in result.columns
    assert all(result['RA'] >= 0)  # RA should be non-negative
    assert all(result['RA'] <= 1)  # RA should be <= 1

def test_M10(sample_data):
    result = M10(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two days of data
    assert 'M10' in result.columns
    assert 'M10_start' in result.columns
    assert all(result['M10'] >= 0)  # M10 should be non-negative
    assert all(result['M10_start'].between(0, 23))  # Hour should be between 0-23

def test_L5(sample_data):
    result = L5(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two days of data
    assert 'L5' in result.columns
    assert 'L5_start' in result.columns
    assert all(result['L5'] >= 0)  # L5 should be non-negative
    assert all(result['L5_start'].between(0, 23))  # Hour should be between 0-23

def test_single_day_data():
    # Create one day of data
    dates = pd.date_range(
        start='2024-01-01', 
        end='2024-01-01 23:59:00', 
        freq='1min'
    )
    activity = np.random.normal(5, 2, size=len(dates))
    data = pd.DataFrame({'ENMO': activity}, index=dates)
    
    # Test all functions with single day data
    for func in [IV, IS, RA, M10, L5]:
        result = func(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

def test_empty_data():
    empty_data = pd.DataFrame({'ENMO': []}, index=pd.DatetimeIndex([]))
    
    # Test IV with empty data
    iv_result = IV(empty_data)
    assert isinstance(iv_result, pd.DataFrame)
    assert len(iv_result) == 0

    # Test IS with empty data
    is_result = IS(empty_data)
    assert isinstance(is_result, pd.DataFrame)
    assert len(is_result) == 0

    # Test RA with empty data
    ra_result = RA(empty_data)
    assert isinstance(ra_result, pd.DataFrame)
    assert len(ra_result) == 0

    # Test M10 with empty data
    m10_result = M10(empty_data)
    assert isinstance(m10_result, pd.DataFrame)
    assert len(m10_result) == 0

    # Test L5 with empty data
    l5_result = L5(empty_data)
    assert isinstance(l5_result, pd.DataFrame)
    assert len(l5_result) == 0

def test_constant_data():
    # Create data with constant values
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-02 23:59:00',
        freq='1min'
    )
    activity = np.ones(len(dates))
    data = pd.DataFrame({'ENMO': activity}, index=dates)

    # Test IV with constant data
    iv_result = IV(data)
    assert isinstance(iv_result, pd.DataFrame)
    assert 'IV' in iv_result.columns
    assert all(iv_result['IV'].isna())  # IV should be NaN for constant data

    # Test IS with constant data
    is_result = IS(data)
    assert isinstance(is_result, pd.DataFrame)
    assert 'IS' in is_result.columns
    assert all(is_result['IS'] == 0)  # IS should be 0 for constant data

    # Test RA with constant data
    ra_result = RA(data)
    assert isinstance(ra_result, pd.DataFrame)
    assert 'RA' in ra_result.columns
    assert all(ra_result['RA'] == 0)  # RA should be 0 for constant data

    # Test M10 with constant data
    m10_result = M10(data)
    assert isinstance(m10_result, pd.DataFrame)
    assert 'M10' in m10_result.columns
    assert all(m10_result['M10'] == 1)  # M10 should equal the constant value

    # Test L5 with constant data
    l5_result = L5(data)
    assert isinstance(l5_result, pd.DataFrame)
    assert 'L5' in l5_result.columns
    assert all(l5_result['L5'] == 1)  # L5 should equal the constant value