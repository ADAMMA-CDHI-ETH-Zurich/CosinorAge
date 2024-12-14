# test_sleep_metrics.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cosinorage.features.utils.sleep_metrics import (
    apply_sleep_wake_predictions,
    WASO,
    TST,
    PTA,
    NWB,
    SOL,
    SRI
)

@pytest.fixture
def sample_data():
    # Create 2 days of minute-by-minute data
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-02 23:59:00',
        freq='1min'
    )
    
    df = pd.DataFrame(index=dates)
    df['ENMO'] = np.random.random(len(dates)) * 0.1  # Random ENMO values
    return df

@pytest.fixture
def sleep_data():
    # Create 2 days of sleep data (1=sleep, 0=wake)
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-02 23:59:00',
        freq='1min'
    )
    
    df = pd.DataFrame(index=dates)
    # Initialize all as wake (0)
    df['sleep'] = 0
    
    # Create a more realistic sleep pattern:
    # Sleep from 23:00 to 07:00 (8 hours of sleep)
    for date in pd.unique(dates.date):
        # Night sleep (from 23:00 to midnight)
        night_mask = (
            (df.index.date == date) & 
            (df.index.hour >= 23)
        )
        # Morning sleep (from midnight to 07:00 of next day)
        next_date = pd.Timestamp(date) + pd.Timedelta(days=1)
        morning_mask = (
            (df.index.date == next_date.date()) & 
            (df.index.hour < 7)
        )
        
        df.loc[night_mask, 'sleep'] = 1
        df.loc[morning_mask, 'sleep'] = 1
    
    return df

def test_apply_sleep_wake_predictions(sample_data):
    sleep_params = {"sleep_ck_sf": 0.0025, "sleep_rescore": True}
    result = apply_sleep_wake_predictions(sample_data, sleep_params)
    
    assert isinstance(result, pd.Series)
    assert set(result.unique()).issubset({0, 1})
    assert len(result) == len(sample_data)

def test_apply_sleep_wake_predictions_missing_column():
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    sleep_params = {"sleep_ck_sf": 0.0025, "sleep_rescore": True}
    
    with pytest.raises(ValueError):
        apply_sleep_wake_predictions(df, sleep_params)

def test_waso(sleep_data):
    result = WASO(sleep_data)
    
    assert isinstance(result, list)
    assert len(result) == 2  # Two days of data
    assert all(isinstance(x, int) for x in result)
    assert all(x >= 0 for x in result)  # WASO should be non-negative

def test_tst(sleep_data):
    result = TST(sleep_data)
    
    assert isinstance(result, list)
    assert len(result) == 2  # Two days of data
    assert all(isinstance(x, int) for x in result)
    assert all(0 <= x <= 1440 for x in result)  # TST should be between 0 and 1440 minutes

def test_pta(sleep_data):
    result = PTA(sleep_data)
    
    assert isinstance(result, list)
    assert len(result) == 2  # Two days of data
    assert all(isinstance(x, float) for x in result)
    # PTA should be between 0 and 1 (representing 0-100%)
    assert all(0 <= x <= 100 for x in result)

def test_nwb(sleep_data):
    result = NWB(sleep_data)
    
    assert isinstance(result, list)
    assert len(result) == 2  # Two days of data
    assert all(isinstance(x, int) for x in result)
    assert all(x >= 0 for x in result)  # NWB should be non-negative

def test_sol(sleep_data):
    result = SOL(sleep_data)
    
    assert isinstance(result, list)
    assert len(result) == 2  # Two days of data
    assert all(isinstance(x, int) for x in result)
    assert all(x >= 0 for x in result)  # SOL should be non-negative

def test_sri(sleep_data):
    result = SRI(sleep_data)
    
    assert isinstance(result, float)
    assert -100 <= result <= 100  # SRI should be between -100 and 100

def test_sri_empty_data():
    empty_df = pd.DataFrame()
    result = SRI(empty_df)
    
    assert np.isnan(result)

def test_sri_insufficient_data():
    # Create less than 2 days of data
    dates = pd.date_range(start='2024-01-01', end='2024-01-01 23:59:00', freq='1min')
    df = pd.DataFrame(index=dates)
    df['sleep'] = 0
    
    result = SRI(df)
    
    assert np.isnan(result)
