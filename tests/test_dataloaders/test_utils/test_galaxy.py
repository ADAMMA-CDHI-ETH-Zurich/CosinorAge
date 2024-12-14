import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from cosinorage.datahandlers.utils.galaxy import (
    read_galaxy_data,
    filter_galaxy_data,
    resample_galaxy_data,
    preprocess_galaxy_data,
    acceleration_data_to_dataframe,
    calibrate,
    remove_noise,
    detect_wear,
    calc_weartime
)

@pytest.fixture
def sample_acc_data():
    """Create sample accelerometer data for testing"""
    # Create 5 days of data at 25Hz (40ms intervals)
    dates = pd.date_range(start='2024-01-01', end='2024-01-06', freq='40ms')
    n_samples = len(dates)
    
    # Create realistic accelerometer data with gravity component
    t = np.linspace(0, n_samples/25, n_samples)  # time vector at 25Hz
    gravity = 9.81  # m/s^2
    
    # Simulate movement with gravity and noise
    df = pd.DataFrame({
        'X': np.sin(2*np.pi*0.5*t) * 0.1 * gravity + np.random.normal(0, 0.01*gravity, n_samples),
        'Y': np.cos(2*np.pi*0.5*t) * 0.1 * gravity + np.random.normal(0, 0.01*gravity, n_samples),
        'Z': np.ones(n_samples) * gravity + np.random.normal(0, 0.01*gravity, n_samples)
    }, index=dates)
    
    return df

@pytest.fixture
def mock_binary_data():
    """Create mock binary acceleration data"""
    class MockSample:
        def __init__(self, x, y, z, timestamp):
            self.acceleration_x = x
            self.acceleration_y = y
            self.acceleration_z = z
            self.unix_timestamp_in_ms = timestamp
            self.sensor_body_location = "WRIST"
            self.effective_time_frame = 1000

    class MockData:
        def __init__(self):
            self.samples = [
                MockSample(1.0, 2.0, 3.0, 1704067200000),
                MockSample(1.1, 2.1, 3.1, 1704067240000)
            ]
    
    return MockData()

def test_read_galaxy_data(tmp_path):
    """Test reading Galaxy Watch data files"""
    # Create temporary directory structure with mock data
    day_dir = tmp_path / "day1"
    day_dir.mkdir()
    mock_file = day_dir / "acceleration_data_1.binary"
    mock_file.write_text("mock_data")  # Just create an empty file for testing

    with patch('cosinorage.datahandlers.utils.galaxy.load_acceleration_data') as mock_load:
        with patch('cosinorage.datahandlers.utils.galaxy.acceleration_data_to_dataframe') as mock_convert:
            # Setup mock returns
            mock_load.return_value = "mock_binary_data"
            mock_convert.return_value = pd.DataFrame({
                'unix_timestamp_in_ms': [1704067200000],
                'acceleration_x': [1.0],
                'acceleration_y': [2.0],
                'acceleration_z': [3.0],
                'sensor_body_location': ['WRIST'],
                'effective_time_frame': [1000]
            })

            meta_dict = {}
            result = read_galaxy_data(str(tmp_path) + "/", meta_dict, verbose=True)

            assert isinstance(result, pd.DataFrame)
            assert all(col in result.columns for col in ['X', 'Y', 'Z'])
            assert 'raw_n_datapoints' in meta_dict

def test_filter_galaxy_data(sample_acc_data):
    """Test filtering Galaxy Watch data"""
    meta_dict = {}
    result = filter_galaxy_data(sample_acc_data, meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= len(sample_acc_data)
    # Check that we have at least 4 consecutive days
    assert len(np.unique(result.index.date)) >= 4

def test_resample_galaxy_data(sample_acc_data):
    """Test resampling Galaxy Watch data"""
    meta_dict = {}
    result = resample_galaxy_data(sample_acc_data, meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    # Check if timestamps are exactly 40ms apart
    time_diffs = np.diff(result.index.astype(np.int64)) / 1e6  # Convert to milliseconds
    assert np.allclose(time_diffs, 40, atol=1)

def test_preprocess_galaxy_data(sample_acc_data, mock_calibrator, mock_wear_detector):
    """Test preprocessing Galaxy Watch data"""
    preprocess_args = {
        'autocalib_sphere_crit': 1,
        'autocalib_sd_criter': 0.3,
        'filter_type': 'highpass',
        'filter_cutoff': 0.5,
        'wear_sd_crit': 0.00013,
        'wear_range_crit': 0.00067,
        'wear_window_length': 30,
        'wear_window_skip': 7
    }
    meta_dict = {}
    
    with patch('cosinorage.datahandlers.utils.galaxy.CalibrateAccelerometer', 
              return_value=mock_calibrator):
        with patch('cosinorage.datahandlers.utils.galaxy.AccelThresholdWearDetection',
                  return_value=mock_wear_detector):
            result = preprocess_galaxy_data(sample_acc_data, preprocess_args, meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert 'ENMO' in result.columns
    assert all(col + '_raw' in result.columns for col in ['X', 'Y', 'Z'])

def test_acceleration_data_to_dataframe(mock_binary_data):
    """Test converting binary acceleration data to DataFrame"""
    result = acceleration_data_to_dataframe(mock_binary_data)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in [
        'acceleration_x', 'acceleration_y', 'acceleration_z',
        'sensor_body_location', 'unix_timestamp_in_ms', 'effective_time_frame'
    ])
    assert len(result) == len(mock_binary_data.samples)

@pytest.fixture
def mock_calibrator():
    """Mock CalibrateAccelerometer for testing"""
    class MockCalibrator:
        def predict(self, time, accel, fs):
            # Return mock calibrated data
            return {
                'accel': accel,  # Just return input data for testing
                'offset': np.array([0.1, 0.1, 0.1]),
                'scale': np.array([1.1, 1.1, 1.1])
            }
    return MockCalibrator()

@pytest.fixture
def mock_wear_detector():
    """Mock AccelThresholdWearDetection for testing"""
    class MockWearDetector:
        def predict(self, time, accel, fs):
            # Return mock wear periods
            return {
                'wear': np.array([[0, len(time)-1]])  # Mark all data as wear
            }
    return MockWearDetector()

def test_calibrate(sample_acc_data, mock_calibrator):
    """Test accelerometer calibration"""
    meta_dict = {}
    
    with patch('cosinorage.datahandlers.utils.galaxy.CalibrateAccelerometer', 
              return_value=mock_calibrator):
        result = calibrate(sample_acc_data, sf=25, sphere_crit=1, sd_criteria=0.3, 
                          meta_dict=meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['X', 'Y', 'Z'])
    assert 'calibration_offset' in meta_dict
    assert 'calibration_scale' in meta_dict

def test_remove_noise(sample_acc_data):
    """Test noise removal from accelerometer data"""
    # Test lowpass filter
    result_lowpass = remove_noise(sample_acc_data, sf=25, filter_type='lowpass', 
                                filter_cutoff=2, verbose=True)
    assert isinstance(result_lowpass, pd.DataFrame)
    
    # Test highpass filter
    result_highpass = remove_noise(sample_acc_data, sf=25, filter_type='highpass', 
                                 filter_cutoff=0.5, verbose=True)
    assert isinstance(result_highpass, pd.DataFrame)
    
    # Test invalid filter type
    with pytest.raises(ValueError):
        remove_noise(sample_acc_data, sf=25, filter_type='bandpass', 
                    filter_cutoff=2, verbose=True)

def test_detect_wear(sample_acc_data, mock_wear_detector):
    """Test wear detection"""
    meta_dict = {}
    
    with patch('cosinorage.datahandlers.utils.galaxy.AccelThresholdWearDetection',
              return_value=mock_wear_detector):
        result = detect_wear(sample_acc_data, sf=25, sd_crit=0.00013, range_crit=0.00067,
                            window_length=30, window_skip=7, meta_dict=meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert set(result['wear'].unique()).issubset({0, 1})

def test_calc_weartime(sample_acc_data):
    """Test wear time calculation"""
    # Add wear column to sample data
    sample_acc_data['wear'] = np.random.choice([0, 1], size=len(sample_acc_data))
    
    meta_dict = {}
    calc_weartime(sample_acc_data, sf=25, meta_dict=meta_dict, verbose=True)
    
    assert 'total_time' in meta_dict
    assert 'wear_time' in meta_dict
    assert 'non-wear_time' in meta_dict
    assert meta_dict['total_time'] > 0
    assert meta_dict['wear_time'] >= 0
    assert meta_dict['non-wear_time'] >= 0
    assert np.isclose(meta_dict['total_time'], 
                     meta_dict['wear_time'] + meta_dict['non-wear_time'])