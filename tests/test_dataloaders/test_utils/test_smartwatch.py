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


def test_calc_weartime():
    # Test case 1: Basic test with 1Hz sampling
    timestamps_1hz = pd.date_range(start='2024-01-01', periods=10, freq='1s')
    df_1hz = pd.DataFrame({
        'wear': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    }, index=timestamps_1hz)
    
    total, wear, nonwear = calc_weartime(df_1hz, sf=1)
    assert total == 9.0  # 9 seconds between first and last timestamp
    assert wear == 7.0   # 7 samples with wear=1
    assert nonwear == 2.0  # total - wear time
    assert abs((total - (wear + nonwear))) < 1e-10  # times should sum to total

    # Test case 2: Test with 2Hz sampling frequency
    timestamps_2hz = pd.date_range(start='2024-01-01', periods=10, freq='500ms')
    df_2hz = pd.DataFrame({
        'wear': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    }, index=timestamps_2hz)
    
    total, wear, nonwear = calc_weartime(df_2hz, sf=2)
    assert total == 4.5  # 4.5 seconds between first and last timestamp
    assert wear == 3.5   # 7 samples * 0.5 seconds each
    assert nonwear == 1.0  # 2 samples * 0.5 seconds each
    assert abs((total - (wear + nonwear))) < 1e-10

    # Test case 3: Empty DataFrame
    empty_df = pd.DataFrame({'wear': []}, index=pd.DatetimeIndex([]))
    with pytest.raises(IndexError):
        calc_weartime(empty_df, sf=1)

    # Test case 4: Single sample
    timestamp_single = pd.date_range(start='2024-01-01', periods=1)
    df_single = pd.DataFrame({'wear': [1]}, index=timestamp_single)
    
    total, wear, nonwear = calc_weartime(df_single, sf=1)
    assert total == 0.0  # No time difference with single sample
    assert wear == 1.0   # One wear sample
    assert nonwear == -1.0  # total - wear


def test_detect_wear():
    # Test 1: Valid input with wear/non-wear periods
    # Create mock data: First 30s movement (wear), last 30s no movement (non-wear)
    # Test parameters
    sampling_freq = 50  # Hz
    duration = 60*60*24  # seconds
    timestamps = pd.date_range(
        start='2024-01-01', 
        periods=duration * sampling_freq, 
        freq=f'{1000/sampling_freq}ms'
    )

    n_samples = len(timestamps)
    mid_point = n_samples // 2
    
    movement_data = np.random.normal(loc=0, scale=0.1, size=(mid_point, 3))
    no_movement_data = np.zeros((n_samples - mid_point, 3))
    acc_data = np.vstack([movement_data, no_movement_data])
    
    df = pd.DataFrame(
        acc_data, 
        columns=['X', 'Y', 'Z'], 
        index=timestamps
    )
    
    result = detect_wear(df, sampling_freq)
    
    # Verify basic properties
    assert isinstance(result, pd.DataFrame)
    assert 'wear' in result.columns
    assert len(result) > 0
    assert result['wear'].between(0, 1).all()
    
    # Verify wear detection accuracy
    mid_time = timestamps[mid_point]
    wear_period = result.loc[:mid_time]['wear'].mean()
    non_wear_period = result.loc[mid_time:]['wear'].mean()
    assert wear_period > 0.7  # First half should be mostly wear
    assert non_wear_period < 0.3  # Second half should be mostly non-wear

    # Test 2: Valid input with wear/non-wear periods
    # Create mock data: First 30s movement (wear), last 30s no movement (non-wear)
    # Test parameters
    sampling_freq = 50  # Hz
    duration = 60  # seconds
    timestamps = pd.date_range(
        start='2024-01-01', 
        periods=duration * sampling_freq, 
        freq=f'{1000/sampling_freq}ms'
    )
    
    n_samples = len(timestamps)
    mid_point = n_samples // 2
    
    movement_data = np.random.normal(loc=0, scale=0.1, size=(mid_point, 3))
    no_movement_data = np.zeros((n_samples - mid_point, 3))
    acc_data = np.vstack([movement_data, no_movement_data])
    
    df = pd.DataFrame(
        acc_data, 
        columns=['X', 'Y', 'Z'], 
        index=timestamps
    )
    
    with pytest.raises(ValueError):
        detect_wear(df, sampling_freq)
    
    # Test 2: Invalid input - missing columns
    df_missing_cols = pd.DataFrame({
        'X': [1, 2, 3],
        'Y': [1, 2, 3]
    })
    with pytest.raises(ValueError):
        detect_wear(df_missing_cols, 50)
    
    # Test 3: Invalid input - empty DataFrame
    df_empty = pd.DataFrame(columns=['X', 'Y', 'Z'])
    with pytest.raises(ValueError):
        detect_wear(df_empty, 50)


def test_remove_noise():
    # Create sample data
    sample_size = 1000
    time_index = pd.date_range(start='2023-01-01', periods=sample_size, freq='12.5ms')
    
    # Generate noisy sine waves for X, Y, Z
    t = np.linspace(0, 10, sample_size)
    noise = np.random.normal(0, 0.5, sample_size)
    
    data = {
        'X': np.sin(2 * np.pi * 0.5 * t) + noise,
        'Y': np.sin(2 * np.pi * 0.3 * t + np.pi/4) + noise,
        'Z': np.sin(2 * np.pi * 0.7 * t + np.pi/2) + noise
    }
    
    df = pd.DataFrame(data, index=time_index)
    
    # Apply noise removal
    filtered_df = remove_noise(df, sf=80)
    
    # Assertions
    assert isinstance(filtered_df, pd.DataFrame)
    assert filtered_df.shape == df.shape
    assert all(col in filtered_df.columns for col in ['X', 'Y', 'Z'])

    # Check that filtered data has less variance than original
    for col in ['X', 'Y', 'Z']:
        assert filtered_df[col].var() < df[col].var()
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        remove_noise(pd.DataFrame(), sf=80)  # Empty DataFrame
    
    with pytest.raises(KeyError):
        remove_noise(pd.DataFrame({'A': [1, 2, 3]}), sf=80)  # Missing required columns


def test_auto_calibrate():
    pass