import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from cosinorage.features.utils.physical_activity_metrics import activity_metrics, cutpoints

@pytest.fixture
def sample_data():
    # Create sample data for one day with readings every minute
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='min')[:-1]  # 1440 minutes
    
    # Create ENMO values that fall into different categories
    enmo_values = np.concatenate([
        np.zeros(480),          # 8 hours of SB (below SB cutpoint)
        np.full(480, 0.005),    # 8 hours of LIPA (between SB and LIPA cutpoints)
        np.full(480, 0.02)      # 8 hours of MVPA (above LIPA cutpoint)
    ])
    
    # Return as DataFrame instead of Series
    return pd.DataFrame({
        'ENMO': enmo_values
    }, index=dates)

def test_activity_metrics_basic(sample_data):
    """Test that activity_metrics returns expected format and values."""
    result = activity_metrics(sample_data)
    
    # Check basic properties
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['SB', 'LIPA', 'MVPA']
    assert isinstance(result.index[0], date)

def test_activity_metrics_values(sample_data):
    """Test that activity_metrics calculates correct hours for each category."""
    result = activity_metrics(sample_data)
    
    # We expect 8 hours in each category based on our sample data
    # Convert string date to datetime.date object
    test_date = pd.to_datetime('2024-01-01').date()
    assert result.loc[test_date, 'SB'] == pytest.approx(8.0)
    assert result.loc[test_date, 'LIPA'] == pytest.approx(8.0)
    assert result.loc[test_date, 'MVPA'] == pytest.approx(8.0)

def test_activity_metrics_multiple_days():
    """Test that activity_metrics handles multiple days correctly."""
    # Create two days of data
    dates = pd.date_range(start='2024-01-01', end='2024-01-03', freq='min')[:-1]
    enmo_values = np.tile([0, 0.005, 0.02], len(dates)//3)
    # Return as DataFrame instead of Series
    data = pd.DataFrame({
        'ENMO': enmo_values
    }, index=dates)
    
    result = activity_metrics(data)
    
    assert len(result) == 2  # Should have data for 2 days
    assert all(isinstance(d, date) for d in result.index)

def test_activity_metrics_empty_data():
    """Test that activity_metrics handles empty data appropriately."""
    # Create empty DataFrame with expected structure
    empty_data = pd.DataFrame(
        columns=['ENMO'],
        index=pd.DatetimeIndex([]),
        dtype=float
    )
    
    result = activity_metrics(empty_data)
    
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ['SB', 'LIPA', 'MVPA']  # Verify expected columns exist

def test_activity_metrics_missing_values():
    """Test that activity_metrics handles missing values appropriately."""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='min')
    # Create DataFrame instead of Series
    data = pd.DataFrame({
        'ENMO': [0, 0.005, 0.02, np.nan] * 15
    }, index=dates)
    
    result = activity_metrics(data)
    
    assert not result.isna().any().any()  # No NaN values in results

def test_activity_metrics_boundary_values():
    """Test that activity_metrics correctly handles boundary values."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='min')
    data = pd.DataFrame({
        'ENMO': [
            cutpoints['SB'],           # Should count as SB
            cutpoints['SB'] + 0.00001, # Should count as LIPA
            cutpoints['LIPA'],         # Should count as LIPA
            cutpoints['LIPA'] + 0.001, # Should count as MVPA
            0.0                        # Should count as SB
        ]
    }, index=dates)
    
    result = activity_metrics(data)
    
    # Convert string date to datetime.date object
    test_date = pd.to_datetime('2024-01-01').date()
    # Convert minutes to hours (2/60 ≈ 0.033 hours)
    assert result.loc[test_date, 'SB'] == pytest.approx(2/60)
    assert result.loc[test_date, 'LIPA'] == pytest.approx(2/60)
    assert result.loc[test_date, 'MVPA'] == pytest.approx(1/60)