import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.dataloaders.utils.ukb import read_ukb_data, filter_ukb_data, resample_ukb_data

@pytest.fixture
def sample_qc_data():
    """Create sample QC data for testing"""
    return pd.DataFrame({
        'eid': [1000, 2000],
        'acc_data_problem': ['', 'problem'],
        'acc_weartime': ['Yes', 'No'],
        'acc_calibration': ['Yes', 'No'],
        'acc_owndata': ['Yes', 'No'],
        'acc_interrupt_period': [0, 1]
    })

@pytest.fixture
def sample_enmo_data():
    """Create sample ENMO data for testing with 5 consecutive days"""
    # Create 5 days of minute-level data
    dates = pd.date_range(start='2023-01-01', periods=5*1440, freq='1min')
    data = pd.DataFrame({
        'TIMESTAMP': dates,
        'ENMO': np.random.rand(5*1440) * 100
    }).set_index('TIMESTAMP')
    return data

def test_read_ukb_data_file_not_found():
    """Test handling of non-existent files"""
    with pytest.raises(FileNotFoundError):
        read_ukb_data('nonexistent.csv', 'nonexistent_dir', 1000)

def test_read_ukb_data_invalid_eid(tmp_path, sample_qc_data):
    """Test handling of invalid EID"""
    qc_file = tmp_path / "qc.csv"
    sample_qc_data.to_csv(qc_file)
    enmo_dir = tmp_path / "enmo"
    enmo_dir.mkdir()
    
    with pytest.raises(ValueError, match="Eid .* not found in QA file"):
        read_ukb_data(qc_file, enmo_dir, 9999)

def test_filter_ukb_data(sample_enmo_data):
    """Test filtering of UKB data"""
    filtered_data = filter_ukb_data(sample_enmo_data)
    assert isinstance(filtered_data, pd.DataFrame)
    assert not filtered_data.empty
    assert filtered_data.index.is_monotonic_increasing
    assert len(np.unique(filtered_data.index.date)) >= 4  # Verify at least 4 days of data

def test_resample_ukb_data(sample_enmo_data):
    """Test resampling of UKB data"""
    # Create gaps in the data
    sparse_data = sample_enmo_data.iloc[::2]
    resampled_data = resample_ukb_data(sparse_data)
    
    assert isinstance(resampled_data, pd.DataFrame)
    assert len(resampled_data) > len(sparse_data)
    assert resampled_data.index.freq == pd.Timedelta('1 min')

def test_filter_ukb_data_incomplete_days(sample_enmo_data):
    """Test filtering of incomplete days"""
    # Remove some data but keep enough complete days
    incomplete_data = sample_enmo_data.copy()
    
    # Remove a few hours of data from the middle of day 3
    # This keeps days 1, 2, 4, and 5 complete while making day 3 incomplete
    day3_start = incomplete_data.index[2*1440]  # Start of day 3
    day3_end = incomplete_data.index[3*1440]    # End of day 3
    
    # Drop day 3 completely to maintain consecutive days 1-2 and 4-5
    incomplete_data = incomplete_data.drop(incomplete_data.loc[day3_start:day3_end].index)
    
    with pytest.raises(ValueError, match="Less than 4 consecutive days found"):
        filtered_data = filter_ukb_data(incomplete_data)

def test_filter_ukb_data_with_valid_gap():
    """Test filtering with a valid gap that maintains 4 consecutive days"""
    # Create 6 days of data to allow for a gap while maintaining 4 consecutive days
    dates = pd.date_range(start='2023-01-01', periods=6*1440, freq='1min')
    data = pd.DataFrame({
        'TIMESTAMP': dates,
        'ENMO': np.random.rand(6*1440) * 100
    }).set_index('TIMESTAMP')
    
    # Remove some hours from day 5 but keep days 1-4 complete
    day5_start = data.index[4*1440 + 6*60]  # Start of hour 6 on day 5
    day5_end = data.index[4*1440 + 12*60]   # End of hour 12 on day 5
    
    # Drop 6 hours of data from day 5
    data_with_gap = data.drop(data.loc[day5_start:day5_end].index)
    
    filtered_data = filter_ukb_data(data_with_gap)
    
    # Should keep days 1-4 and filter out days 5-6
    assert len(np.unique(filtered_data.index.date)) == 4
    
    # Verify that the remaining days are consecutive
    remaining_days = np.unique(filtered_data.index.date)
    day_diffs = np.diff(remaining_days)
    assert all(diff.days == 1 for diff in day_diffs)

def test_filter_ukb_data_insufficient_days():
    """Test handling of insufficient consecutive days"""
    # Create data with only 3 days
    dates = pd.date_range(start='2023-01-01', periods=3*1440, freq='1min')
    insufficient_data = pd.DataFrame({
        'TIMESTAMP': dates,
        'ENMO': np.random.rand(3*1440) * 100
    }).set_index('TIMESTAMP')
    
    with pytest.raises(ValueError, match="Less than 4 consecutive days found"):
        filter_ukb_data(insufficient_data)

def test_resample_ukb_data_with_gaps(sample_enmo_data):
    """Test resampling with data gaps"""
    # Create artificial gaps
    data_with_gaps = sample_enmo_data.copy()
    data_with_gaps = data_with_gaps.drop(data_with_gaps.index[10:20])
    
    resampled_data = resample_ukb_data(data_with_gaps)
    
    assert len(resampled_data) == len(sample_enmo_data)
    assert not resampled_data.isnull().any().any()

@pytest.mark.parametrize("verbose", [True, False])
def test_verbose_output(sample_enmo_data, verbose, capsys):
    """Test verbose output option"""
    filter_ukb_data(sample_enmo_data, verbose=verbose)
    captured = capsys.readouterr()
    
    if verbose:
        assert len(captured.out) > 0
    else:
        assert len(captured.out) == 0