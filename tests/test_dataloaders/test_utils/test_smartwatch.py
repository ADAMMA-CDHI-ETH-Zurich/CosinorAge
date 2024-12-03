import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from cosinorage.dataloaders.utils.smartwatch import (
    read_smartwatch_data,
    preprocess_smartwatch_data,
    remove_noise,
    detect_wear,
    calc_weartime
)

@pytest.fixture
def sample_acc_data():
    """Create sample accelerometer data with longer duration"""
    # Create 1 hour of data at 50Hz
    n_samples = 50 * 60 * 60  # 1 hour of data at 50Hz
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='20ms')
    data = pd.DataFrame({
        'X': np.sin(np.linspace(0, 10*np.pi, n_samples)) * 1000,
        'Y': np.cos(np.linspace(0, 10*np.pi, n_samples)) * 1000,
        'Z': np.random.normal(0, 100, n_samples)
    }, index=timestamps)
    return data

@pytest.fixture
def sample_directory(tmp_path):
    """Create temporary directory with sample CSV files"""
    directory = tmp_path / "test_data"
    directory.mkdir()
    
    # Create sample CSV files
    for i in range(2):
        df = pd.DataFrame({
            'HEADER_TIMESTAMP': pd.date_range(start=f'2023-01-0{i+1}', periods=100, freq='20ms'),
            'X': np.random.normal(0, 100, 100),
            'Y': np.random.normal(0, 100, 100),
            'Z': np.random.normal(0, 100, 100)
        })
        df.to_csv(directory / f"file_{i}.sensor.csv", index=False)
    
    return directory

def test_read_smartwatch_data(sample_directory):
    """Test reading smartwatch data from CSV files"""
    meta_dict = {}
    data = read_smartwatch_data(str(sample_directory), meta_dict)
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['X', 'Y', 'Z'])
    assert isinstance(data.index, pd.DatetimeIndex)

def test_remove_noise(sample_acc_data):
    """Test noise removal function"""
    filtered_data = remove_noise(sample_acc_data, sf=50, filter_type='lowpass', filter_cutoff=2)
    
    assert isinstance(filtered_data, pd.DataFrame)
    assert filtered_data.shape == sample_acc_data.shape
    assert all(col in filtered_data.columns for col in ['X', 'Y', 'Z'])
    
    # Test invalid filter type
    with pytest.raises(ValueError):
        remove_noise(sample_acc_data, sf=50, filter_type='bandpass', filter_cutoff=2)

def test_detect_wear(sample_acc_data):
    """Test wear detection"""
    meta_dict = {}
    wear_data = detect_wear(sample_acc_data, sf=50, meta_dict=meta_dict)
    
    assert isinstance(wear_data, pd.DataFrame)
    assert 'wear' in wear_data.columns
    assert wear_data['wear'].isin([0, 1]).all()

def test_calc_weartime(sample_acc_data):
    """Test wear time calculation"""
    meta_dict = {}
    sample_acc_data['wear'] = np.random.choice([0, 1], size=len(sample_acc_data))
    
    total, wear, nonwear = calc_weartime(sample_acc_data, sf=50, meta_dict=meta_dict, verbose=False)
    
    assert isinstance(total, float)
    assert isinstance(wear, float)
    assert isinstance(nonwear, float)
    assert total >= wear
    assert total >= nonwear
    assert abs(total - (wear + nonwear)) < 1e-10

def test_preprocess_smartwatch_data(sample_acc_data):
    """Test complete preprocessing pipeline"""
    meta_dict = {}
    preprocessed_data = preprocess_smartwatch_data(
        sample_acc_data, 
        sf=50, 
        meta_dict=meta_dict,
        preprocess_args={
            'filter_type': 'lowpass',
            'filter_cutoff': 2,
            'autocalib_sphere_crit': 1,
            'autocalib_sd_criter': 0.3
        }
    )
    
    assert isinstance(preprocessed_data, pd.DataFrame)
    assert all(col in preprocessed_data.columns for col in ['X', 'Y', 'Z', 'wear'])
    assert preprocessed_data['wear'].isin([0, 1]).all()