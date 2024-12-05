import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cosinorage.features.features import WearableFeatures
from cosinorage.datahandlers import DataHandler

@pytest.fixture
def mock_DataHandler():
    # Create mock data with exactly 3 full days of measurements (3 * 1440 minutes)
    dates = pd.date_range(
        start='2024-01-01 00:00:00', 
        end='2024-01-03 23:59:00',  # Include the full last minute of day 3
        freq='1min'
    )
    
    # Create synthetic ENMO data with a daily pattern
    time_of_day = dates.hour + dates.minute/60
    amplitude = 0.5
    baseline = 0.3
    noise = np.random.normal(0, 0.1, len(dates))
    
    # Simulate daily pattern with peak at 15:00 (15 hours)
    enmo = baseline + amplitude * np.sin(2 * np.pi * (time_of_day - 15) / 24) + noise
    enmo = np.maximum(enmo, 0)  # Ensure non-negative values
    
    data = pd.DataFrame({
        'ENMO': enmo
    }, index=dates)
    
    # Verify we have exactly 3 days of data
    assert len(data) == 4320  # 3 days * 1440 minutes
    
    class MockDataHandler(DataHandler):
        def get_ml_data(self):
            return data
            
    return MockDataHandler()

def test_initialization(mock_DataHandler):
    features = WearableFeatures(mock_DataHandler)
    assert isinstance(features.ml_data, pd.DataFrame)
    assert isinstance(features.feature_df, pd.DataFrame)
    assert isinstance(features.feature_dict, dict)
    assert 'ENMO' in features.ml_data.columns

def test_cosinor_features(mock_DataHandler):
    features = WearableFeatures(mock_DataHandler)
    daily_features, multiday_features = features.get_cosinor_features()
    
    # Check daily features
    assert isinstance(daily_features, pd.DataFrame)
    assert all(col in daily_features.columns for col in 
              ['MESOR', 'amplitude', 'acrophase', 'acrophase_time'])
    
    # Check multiday features
    assert isinstance(multiday_features, dict)
    assert all(key in multiday_features for key in 
              ['MESOR', 'amplitude', 'acrophase', 'acrophase_time'])

def test_nonparametric_features(mock_DataHandler):
    features = WearableFeatures(mock_DataHandler)
    
    # Test IV
    iv_data = features.get_IV()
    assert isinstance(iv_data, pd.DataFrame)
    assert 'IV' in iv_data.columns
    
    # Test IS
    is_data = features.get_IS()
    assert isinstance(is_data, pd.DataFrame)
    assert 'IS' in is_data.columns
    
    # Test RA
    ra_data = features.get_RA()
    assert isinstance(ra_data, pd.DataFrame)
    assert 'RA' in ra_data.columns

def test_activity_metrics(mock_DataHandler):
    features = WearableFeatures(mock_DataHandler)
    
    # Test M10 and L5
    m10_data = features.get_M10()
    l5_data = features.get_L5()
    m10_start = features.get_M10_start()
    l5_start = features.get_L5_start()
    
    assert isinstance(m10_data, pd.DataFrame)
    assert isinstance(l5_data, pd.DataFrame)
    assert isinstance(m10_start, pd.DataFrame)
    assert isinstance(l5_start, pd.DataFrame)
    
    # Test activity levels
    sb_data = features.get_SB()
    lipa_data = features.get_LIPA()
    mvpa_data = features.get_MVPA()
    
    assert isinstance(sb_data, pd.DataFrame)
    assert isinstance(lipa_data, pd.DataFrame)
    assert isinstance(mvpa_data, pd.DataFrame)

def test_sleep_metrics(mock_DataHandler):
    features = WearableFeatures(mock_DataHandler)
    
    # Test sleep predictions
    sleep_pred = features.get_sleep_predictions()
    assert isinstance(sleep_pred, pd.DataFrame)
    
    # Test TST
    tst_data = features.get_TST()
    assert isinstance(tst_data, pd.DataFrame)
    
    # Test WASO
    waso_data = features.get_WASO()
    assert isinstance(waso_data, pd.DataFrame)
    
    # Test PTA
    pta_data = features.get_PTA()
    assert isinstance(pta_data, pd.DataFrame)
    
    # Test SRI
    sri_data = features.get_SRI()
    assert isinstance(sri_data, pd.DataFrame)

def test_run_all_features(mock_DataHandler):
    features = WearableFeatures(mock_DataHandler)
    features.run()
    
    feature_df, feature_dict = features.get_all()
    
    # Check if all features are computed
    expected_columns = [
        'MESOR', 'amplitude', 'acrophase', 'acrophase_time',
        'IV', 'IS', 'RA', 'M10', 'L5', 'M10_start', 'L5_start',
        'SB', 'LIPA', 'MVPA', 'TST', 'WASO', 'PTA', 'SRI'
    ]
    
    assert all(col in feature_df.columns for col in expected_columns)
    assert all(key in feature_dict for key in ['MESOR', 'amplitude', 'acrophase', 'acrophase_time'])