import pytest
import numpy as np
import pandas as pd
from cosinorage.features.utils.rescaling import min_max_scaling_exclude_outliers

@pytest.fixture
def sample_data():
    return pd.Series([0, 1, 2, 3, 4, 5, 100])  # Last value is an outlier

def test_basic_scaling(sample_data):
    """Test basic scaling functionality"""
    result = min_max_scaling_exclude_outliers(sample_data)
    assert len(result) == len(sample_data)
    assert isinstance(result, pd.Series)
    assert result.min() == 0
    
def test_numpy_input():
    """Test function works with numpy array input"""
    np_data = np.array([0, 1, 2, 3, 4, 5, 100])
    result = min_max_scaling_exclude_outliers(np_data)
    assert isinstance(result, pd.Series)
    assert len(result) == len(np_data)

def test_outlier_handling(sample_data):
    """Test that outliers are properly handled"""
    result = min_max_scaling_exclude_outliers(sample_data, upper_quantile=0.8)
    # The outlier (100) should be scaled above 100 in the result
    assert result.iloc[-1] > 100

def test_zero_variance():
    """Test handling of constant values"""
    constant_data = pd.Series([5, 5, 5, 5])
    result = min_max_scaling_exclude_outliers(constant_data)
    assert all(result == 0)  # All values should be scaled to 0

def test_negative_values():
    """Test scaling with negative values"""
    negative_data = pd.Series([-10, -5, 0, 5, 10, 100])
    result = min_max_scaling_exclude_outliers(negative_data)
    assert result.min() == 0
    assert 0 <= result.median() <= 100

def test_empty_input():
    """Test handling of empty input"""
    empty_data = pd.Series([])
    with pytest.raises(ValueError):
        min_max_scaling_exclude_outliers(empty_data)

def test_single_value():
    """Test scaling with single value"""
    single_value = pd.Series([42])
    result = min_max_scaling_exclude_outliers(single_value)
    assert result[0] == 0  # Single value should be scaled to 0