import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cosinorage.dataloaders.utils.smartwatch import *

# Test basic helper functions first
def test_roll_mean():
    # Test case 1: Basic sequence with known means
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = roll_mean(data, window_size=3)
    expected = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 2: Sequence with negative numbers and decimals
    data = np.array([-1.5, 2.7, -3.2, 4.1, -5.9, 6.3, -7.8, 8.2])
    result = roll_mean(data, window_size=4)
    expected = np.array([0.525, -0.575, 0.325, -0.825, 0.2])  # Means of sliding windows
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 3: Constant sequence (should return same value)
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    result = roll_mean(data, window_size=3)
    expected = np.array([5.0, 5.0, 5.0, 5.0])
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 4: Different window sizes
    data = np.array([1, 2, 3, 4, 5])
    result_w2 = roll_mean(data, window_size=2)
    result_w3 = roll_mean(data, window_size=3)
    result_w4 = roll_mean(data, window_size=4)
    np.testing.assert_array_almost_equal(result_w2, np.array([1.5, 2.5, 3.5, 4.5]))
    np.testing.assert_array_almost_equal(result_w3, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_almost_equal(result_w4, np.array([2.5, 3.5]))

    # Test case 5: Large numbers
    data = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    result = roll_mean(data, window_size=3)
    expected = np.array([2e6, 3e6, 4e6])
    np.testing.assert_array_almost_equal(result, expected)

    # Test case 6: Sinusoidal data (common in accelerometer signals)
    t = np.linspace(0, 2*np.pi, 100)
    data = np.sin(t)
    result = roll_mean(data, window_size=10)
    # The mean of a sliding window of sine wave should have smaller amplitude
    assert np.max(np.abs(result)) < np.max(np.abs(data))
    assert len(result) == len(data) - 9

    # Test case 7: Random data with known mean
    np.random.seed(42)
    data = np.random.normal(loc=5, scale=1, size=1000)
    result = roll_mean(data, window_size=100)
    # Mean should be close to 5 for large windows
    assert np.abs(np.mean(result) - 5) < 0.1
    assert len(result) == len(data) - 99

    # Test edge cases
    with pytest.raises(ValueError):
        roll_mean(np.array([1, 2]), window_size=3)  # Array smaller than window
    with pytest.raises(ValueError):
        roll_mean(np.array([1, 2, 3]), window_size=0)  # Invalid window size
    with pytest.raises(ValueError):
        roll_mean(np.array([]), window_size=1)  # Empty array


def test_roll_sd():
    # Test case 1: Basic sequence with known standard deviations
    data = np.array([1, 1, 1, 2, 2, 2])
    result = roll_sd(data, window_size=3)
    expected = np.array([0, 0.57735027, 0.57735027, 0])  # SD of sliding windows
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case 2: Sequence with negative numbers and decimals
    data = np.array([-1.5, 2.7, -3.2, 4.1, -5.9, 6.3])
    result = roll_sd(data, window_size=3)
    expected = np.array([3.0369941, 3.8742741, 5.173329, 6.5023073])  # Verified SDs
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Test case 3: Constant sequence (should return zeros)
    data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    result = roll_sd(data, window_size=3)
    np.testing.assert_array_equal(result, np.zeros(3))

    # Test case 4: Different window sizes
    data = np.array([1, 2, 3, 4, 5])
    result_w2 = roll_sd(data, window_size=2)
    np.testing.assert_array_almost_equal(result_w2, np.array([0.70710678, 0.70710678, 0.70710678, 0.70710678]))

    # Test case 5: Random data with known properties
    np.random.seed(42)
    data = np.random.normal(loc=5, scale=1, size=1000)
    result = roll_sd(data, window_size=100)
    # SD should be close to 1 for large windows
    assert 0.8 < np.mean(result) < 1.2
    assert len(result) == len(data) - 99

    # Test edge cases
    with pytest.raises(ValueError):
        roll_sd(np.array([1, 2]), window_size=3)  # Array smaller than window
    with pytest.raises(ValueError):
        roll_sd(np.array([1, 2, 3]), window_size=0)  # Invalid window size
    with pytest.raises(ValueError):
        roll_sd(np.array([]), window_size=1)  # Empty array
    

def test_sliding_window():
    # Test case 1: Basic sequence with step_size=1
    arr = np.array([1, 2, 3, 4, 5])
    result = sliding_window(arr, window_size=3, step_size=1)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(result, expected)

    # Test case 2: Different step sizes
    result_step2 = sliding_window(arr, window_size=2, step_size=2)
    expected_step2 = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(result_step2, expected_step2)

    # Test case 3: Window size equals array length
    result_full = sliding_window(arr, window_size=5, step_size=1)
    expected_full = np.array([[1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(result_full, expected_full)

    # Test case 4: Float values
    float_arr = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    result_float = sliding_window(float_arr, window_size=3, step_size=2)
    expected_float = np.array([[1.5, 2.7, 3.2], [3.2, 4.8, 5.1]])
    np.testing.assert_array_equal(result_float, expected_float)

    # Test edge cases
    # Test case 5: 2D array input
    arr_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    with pytest.raises(ValueError):
        sliding_window(arr_2d, window_size=2, step_size=1)

    # Test case 6: Empty array
    with pytest.raises(ValueError):
        sliding_window(np.array([]), window_size=1, step_size=1)

    # Test case 7: Window size larger than array
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=6, step_size=1)

    # Test case 8: Invalid step size
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=3, step_size=0)

    # Test case 9: Negative step size
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=3, step_size=-1)

    # Test case 10: Step size larger than array
    with pytest.raises(ValueError):
        sliding_window(arr, window_size=2, step_size=6)


def test_resample_index():
    # Test case 1: Basic functionality with regular intervals
    index = pd.date_range(start='2024-01-01', periods=100, freq='1s')
    result = resample_index(index, window_samples=30, step_samples=10)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['start', 'end'])
    assert len(result) == (100 - 30) // 10 + 1
    
    # Test case 2: Verify window sizes and steps
    assert (result['end'] - result['start']).iloc[0] == pd.Timedelta(seconds=29)  # 30 samples - 1
    assert (result['start'].iloc[1] - result['start'].iloc[0]) == pd.Timedelta(seconds=10)
    
    # Test case 3: Different frequencies
    index_ms = pd.date_range(start='2024-01-01', periods=100, freq='100ms')
    result_ms = resample_index(index_ms, window_samples=20, step_samples=5)
    assert len(result_ms) == (100 - 20) // 5 + 1
    assert (result_ms['end'] - result_ms['start']).iloc[0] == pd.Timedelta(milliseconds=1900)
    
    # Test case 4: Edge case - window_samples equals length of index
    result_edge = resample_index(index, window_samples=100, step_samples=10)
    assert len(result_edge) == 1
    assert result_edge['start'].iloc[0] == index[0]
    assert result_edge['end'].iloc[0] == index[-1]
    
    # Test edge cases and error conditions
    with pytest.raises(ValueError):
        resample_index(index, window_samples=0, step_samples=10)  # Invalid window size
    
    with pytest.raises(ValueError):
        resample_index(index, window_samples=30, step_samples=0)  # Invalid step size
    
    with pytest.raises(ValueError):
        resample_index(index, window_samples=101, step_samples=10)  # Window larger than index
    
    with pytest.raises(ValueError):
        resample_index(pd.DatetimeIndex([]), window_samples=10, step_samples=5)  # Empty index


def testrescore_wear_detection():
    # Create sample wear data with alternating wear/non-wear periods
    wear_data = pd.DataFrame({
        'wear': [1, 1, 1, 0, 0, 1, 1],
        'start': pd.date_range(start='2024-01-01', periods=7, freq='15min'),
        'end': pd.date_range(start='2024-01-01', periods=7, freq='15min') + pd.Timedelta(minutes=15)
    })
    
    result = rescore_wear_detection(wear_data)
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert set(result['wear'].unique()).issubset({0, 1})


"""
# Test basic processing functions
def test_remove_noise():
    # Create noisy signal
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * 0.5 * t)
    noise = np.random.normal(0, 0.2, 1000)
    noisy_signal = clean_signal + noise
    
    df = pd.DataFrame({
        'X': noisy_signal,
        'Y': noisy_signal,
        'Z': noisy_signal
    })
    
    result = remove_noise(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert result.index.equals(df.index)
    assert not np.array_equal(result['X'].values, df['X'].values)
    assert result['X'].std() <= df['X'].std()

def test_auto_calibrate():
    # Create sample data with known offset and scale
    df = pd.DataFrame({
        'X': np.random.normal(0.5, 0.1, 1000),  # offset of 0.5
        'Y': np.random.normal(0, 0.2, 1000) * 2,  # scale of 2
        'Z': np.random.normal(-0.3, 0.1, 1000)  # offset of -0.3
    })
    
    result = auto_calibrate(df, sf=50)
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['X', 'Y', 'Z'])
    assert abs(result['X'].mean()) < abs(df['X'].mean())
    assert abs(result['Z'].mean()) < abs(df['Z'].mean())

def test_detect_wear():
    # Create sample data with periods of movement and non-movement
    index = pd.date_range(start='2024-01-01', periods=5000, freq='20ms')
    movement = np.concatenate([
        np.random.normal(0, 0.5, 2000),  # movement
        np.random.normal(0, 0.01, 1000),  # non-movement
        np.random.normal(0, 0.5, 2000)   # movement
    ])
    
    df = pd.DataFrame({
        'X': movement,
        'Y': movement,
        'Z': movement
    }, index=index)
    
    result = detect_wear(df, sf=50)
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert set(result['wear'].unique()).issubset({0, 1})

def test_calc_weartime():
    # Create sample data
    index = pd.date_range(start='2024-01-01', periods=1000, freq='20ms')
    df = pd.DataFrame({
        'wear': np.concatenate([
            np.ones(500),
            np.zeros(200),
            np.ones(300)
        ])
    }, index=index)
    
    total, wear, nonwear = calc_weartime(df, sf=50)
    assert isinstance(total, float)
    assert isinstance(wear, float)
    assert isinstance(nonwear, float)
    assert abs((wear + nonwear) - total) < 1e-10  # Should sum to total

# Test high-level functions
def test_read_smartwatch_data(tmp_path):
    # Create temporary test CSV files
    df1 = pd.DataFrame({
        'HEADER_TIMESTAMP': pd.date_range(start='2024-01-01', periods=100, freq='20ms'),
        'X': np.random.normal(0, 1, 100),
        'Y': np.random.normal(0, 1, 100),
        'Z': np.random.normal(0, 1, 100)
    })
    df2 = pd.DataFrame({
        'HEADER_TIMESTAMP': pd.date_range(start='2024-01-01 00:00:02', periods=100, freq='20ms'),
        'X': np.random.normal(0, 1, 100),
        'Y': np.random.normal(0, 1, 100),
        'Z': np.random.normal(0, 1, 100)
    })
    
    df1.to_csv(tmp_path / "file1.sensor.csv", index=False)
    df2.to_csv(tmp_path / "file2.sensor.csv", index=False)
    
    # Test successful read
    data, freq = read_smartwatch_data(tmp_path)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(freq, float)
    assert len(data) == 200
    assert all(col in data.columns for col in ['X', 'Y', 'Z'])
    
    # Test empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    data, freq = read_smartwatch_data(empty_dir)
    assert data.empty
    assert freq is None

def test_preprocess_smartwatch_data():
    # Create sample data
    index = pd.date_range(start='2024-01-01', periods=1000, freq='20ms')
    df = pd.DataFrame({
        'X': np.sin(np.linspace(0, 10*np.pi, 1000)),
        'Y': np.cos(np.linspace(0, 10*np.pi, 1000)),
        'Z': np.random.normal(0, 0.1, 1000)
    }, index=index)
    
    meta_dict = {}
    result = preprocess_smartwatch_data(df, sf=50, meta_dict=meta_dict, verbose=True)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['X', 'Y', 'Z', 'wear'])
    assert all(key in meta_dict for key in ['total time', 'wear time', 'non-wear time'])
"""