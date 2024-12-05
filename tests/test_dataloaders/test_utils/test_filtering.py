import pandas as pd
import pytest
from datetime import datetime, timedelta
from cosinorage.datahandlers.utils.filtering import filter_incomplete_days

@pytest.fixture
def sample_data():
    # Create sample data with 1-second frequency (86400 points per day)
    dates = []
    current = datetime(2024, 1, 1, 0, 0, 0)
    
    # Day 1: Complete day (86400 points)
    dates.extend([current + timedelta(seconds=i) for i in range(86400)])
    
    # Day 2: Complete day (86400 points)
    current = datetime(2024, 1, 2, 0, 0, 0)
    dates.extend([current + timedelta(seconds=i) for i in range(86400)])
    
    # Day 3: Incomplete day (43200 points - half day)
    current = datetime(2024, 1, 3, 0, 0, 0)
    dates.extend([current + timedelta(seconds=i) for i in range(43200)])
    
    df = pd.DataFrame(index=dates)
    df['value'] = 1
    return df

def test_complete_days_only():
    """Test that only complete days are retained"""
    # Create test data with 1-second frequency
    dates = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-01-02 23:59:59',
        freq='s'
    )
    df = pd.DataFrame(index=dates)
    df['value'] = 1
    
    filtered_df = filter_incomplete_days(df, data_freq=1)
    
    # Should keep both complete days
    assert len(filtered_df) == 86400 * 2
    assert filtered_df.index.min().date() == datetime(2024, 1, 1).date()
    assert filtered_df.index.max().date() == datetime(2024, 1, 2).date()

def test_incomplete_days_filtered():
    """Test that incomplete days are properly filtered out"""
    # Create test data with partial days
    dates = pd.date_range(
        start='2024-01-01 12:00:00',  # Start at noon (incomplete day)
        end='2024-01-02 23:59:59',    # Complete day
        freq='s'
    )
    df = pd.DataFrame(index=dates)
    df['value'] = 1
    
    filtered_df = filter_incomplete_days(df, data_freq=1)
    
    # Should only keep the complete day
    assert len(filtered_df) == 86400
    assert filtered_df.index.min().date() == datetime(2024, 1, 2).date()

def test_empty_dataframe():
    """Test handling of empty DataFrame"""
    df = pd.DataFrame()
    filtered_df = filter_incomplete_days(df, data_freq=1)
    assert filtered_df.empty

def test_different_frequencies():
    """Test with different data frequencies"""
    # Create test data with 5-minute frequency
    dates = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-01-02 23:55:00',  # End at 23:55 to get complete days
        freq='5min'
    )
    df = pd.DataFrame(index=dates)
    df['value'] = 1
    
    filtered_df = filter_incomplete_days(df, data_freq=1/300)  # 1/300 Hz = 5 minutes
    
    # Should keep both days (288 points per day)
    assert len(filtered_df) == 288 * 2

def test_single_day():
    """Test with single complete day"""
    dates = pd.date_range(
        start='2024-01-01 00:00:00',
        end='2024-01-01 23:59:59',
        freq='s'
    )
    df = pd.DataFrame(index=dates)
    df['value'] = 1
    
    filtered_df = filter_incomplete_days(df, data_freq=1)
    
    assert len(filtered_df) == 86400
    assert filtered_df.index.min().date() == datetime(2024, 1, 1).date()