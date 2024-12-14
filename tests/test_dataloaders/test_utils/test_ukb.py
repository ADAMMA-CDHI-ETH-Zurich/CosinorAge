import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from cosinorage.datahandlers.utils.ukb import read_ukb_data, filter_ukb_data, resample_ukb_data

@pytest.fixture
def sample_qc_data(tmp_path):
    """Create a sample quality control CSV file."""
    qc_data = pd.DataFrame({
        'eid': [1, 2, 3],
        'acc_data_problem': ['', 'problem', ''],
        'acc_weartime': ['Yes', 'No', 'Yes'],
        'acc_calibration': ['Yes', 'Yes', 'No'],
        'acc_owndata': ['Yes', 'No', 'Yes'],
        'acc_interrupt_period': [0, 1, 0]
    })
    qc_file = tmp_path / "qc_data.csv"
    qc_data.to_csv(qc_file, index=False)
    return str(qc_file)

@pytest.fixture
def sample_enmo_data(tmp_path):
    """Create sample ENMO data files."""
    enmo_dir = tmp_path / "enmo_data"
    enmo_dir.mkdir()
    
    # Create sample ENMO file with proper datetime format
    header = 'acceleration data from 2020-01-01 00:00:00 to 2020-01-07 23:59:00'
    data = pd.DataFrame({
        'eid': [1] * 1440,  # One day worth of minutes
        'enmo_mg': [header] + [str(x * 100) for x in range(1439)]  # Sample ENMO values as strings
    })
    
    file_path = enmo_dir / "OUT_sample.csv"
    data.to_csv(file_path, index=False)
    return str(enmo_dir)

def test_read_ukb_data_invalid_paths():
    """Test read_ukb_data with invalid file paths."""
    with pytest.raises(FileNotFoundError):
        read_ukb_data("nonexistent.csv", "nonexistent_dir", 1)

def test_read_ukb_data_invalid_eid(sample_qc_data, sample_enmo_data):
    """Test read_ukb_data with invalid participant ID."""
    with pytest.raises(ValueError):
        read_ukb_data(sample_qc_data, sample_enmo_data, 999)

def test_filter_ukb_data():
    """Test filter_ukb_data with sample data."""
    # Create sample data with 7 days (more than required 4 days)
    dates = pd.date_range(start='2020-01-01', end='2020-01-07 23:59:00', freq='1min')
    data = pd.DataFrame(
        index=dates,
        data={'ENMO': np.random.random(len(dates))}
    )
    
    # Keep all data points to ensure we have complete days
    filtered_data = filter_ukb_data(data)
    
    # Should have exactly 7 days of data
    unique_days = pd.unique(filtered_data.index.date)
    assert len(unique_days) == 7
    # Verify days are consecutive
    day_diffs = np.diff([d.toordinal() for d in unique_days])
    assert all(diff == 1 for diff in day_diffs)

def test_resample_ukb_data():
    """Test resample_ukb_data with irregular timestamps."""
    # Create sample data with irregular timestamps
    dates = pd.date_range(start='2020-01-01', periods=100, freq='90s')
    data = pd.DataFrame(
        index=dates,
        data={'ENMO': np.random.random(len(dates))}
    )
    
    resampled_data = resample_ukb_data(data)
    
    # Check that data is resampled to 1-minute intervals
    assert resampled_data.index.freq == pd.Timedelta('1min')
    assert isinstance(resampled_data, pd.DataFrame)
    assert not resampled_data.isnull().any().any()

def test_filter_ukb_data_consecutive_days():
    """Test that filter_ukb_data properly handles consecutive days requirement."""
    # Create sample data with 7 consecutive days
    dates = pd.date_range(start='2020-01-01', end='2020-01-07 23:59:00', freq='1min')
    data = pd.DataFrame(
        index=dates,
        data={'ENMO': np.random.random(len(dates))}
    )
    
    filtered_data = filter_ukb_data(data)
    
    # Should have exactly 7 consecutive days
    unique_days = pd.unique(filtered_data.index.date)
    assert len(unique_days) == 7
    # Verify days are consecutive
    day_diffs = np.diff([d.toordinal() for d in unique_days])
    assert all(diff == 1 for diff in day_diffs)

def test_resample_ukb_data_missing_values():
    """Test that resample_ukb_data properly handles missing values."""
    # Create sample data with missing values
    dates = pd.date_range(start='2020-01-01', periods=100, freq='1min')
    data = pd.DataFrame(
        index=dates,
        data={'ENMO': np.random.random(len(dates))}
    )
    data.loc[data.index[10:20], 'ENMO'] = np.nan
    
    resampled_data = resample_ukb_data(data)
    
    # Check that missing values were interpolated
    assert not resampled_data.isnull().any().any()
    assert isinstance(resampled_data, pd.DataFrame)