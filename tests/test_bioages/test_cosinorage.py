import unittest
import numpy as np
import pandas as pd
from cosinorage.bioages.cosinorage import CosinorAge

def create_mock_handler():
    class MockHandler:
        def get_ml_data(self):
            # Create a datetime index for 24 hours with minute intervals
            # Using 'min' instead of deprecated 'T' for minute frequency
            index = pd.date_range(start='2024-01-01', periods=1440, freq='min')
            # Create synthetic activity data with a cosinor pattern
            time = np.linspace(0, 24, 1440)
            activity = 0.5 + 0.3 * np.cos(2 * np.pi * (time - 2) / 24)
            # Create DataFrame with required format
            df = pd.DataFrame({'ENMO': activity}, index=index)
            return df
    return MockHandler()

# Rest of the test code remains the same...

def test_cosinor_age_initialization():
    mock_handler = create_mock_handler()
    records = [{"handler": mock_handler, "age": 30, "gender": "male"}]
    cosinor_age = CosinorAge(records)
    assert isinstance(cosinor_age, CosinorAge)
    assert len(cosinor_age.records) == 1

def test_cosinor_age_computation():
    mock_handler = create_mock_handler()
    records = [{"handler": mock_handler, "age": 30, "gender": "male"}]
    cosinor_age = CosinorAge(records)
    predictions = cosinor_age.get_predictions()
    
    assert 'cosinorage' in predictions[0]
    assert 'cosinorage_advance' in predictions[0]
    assert 'mesor' in predictions[0]
    assert 'amp1' in predictions[0]
    assert 'phi1' in predictions[0]
    assert isinstance(predictions[0]['cosinorage'], float)
    assert predictions[0]['cosinorage'] > 0

def test_gender_specific_models():
    mock_handler = create_mock_handler()
    records = [
        {"handler": mock_handler, "age": 30, "gender": "male"},
        {"handler": mock_handler, "age": 30, "gender": "female"},
        {"handler": mock_handler, "age": 30, "gender": "unknown"}
    ]
    cosinor_age = CosinorAge(records)
    predictions = cosinor_age.get_predictions()
    
    male_pred = next(r for r in predictions if r['gender'] == 'male')
    female_pred = next(r for r in predictions if r['gender'] == 'female')
    unknown_pred = next(r for r in predictions if r['gender'] == 'unknown')
    
    assert male_pred['cosinorage'] != female_pred['cosinorage']
    assert male_pred['cosinorage'] != unknown_pred['cosinorage']

def test_cosinorage_advance_calculation():
    mock_handler = create_mock_handler()
    records = [{"handler": mock_handler, "age": 30, "gender": "male"}]
    cosinor_age = CosinorAge(records)
    predictions = cosinor_age.get_predictions()
    
    expected_advance = predictions[0]['cosinorage'] - predictions[0]['age']
    assert abs(predictions[0]['cosinorage_advance'] - expected_advance) < 1e-10

def test_plot_predictions():
    mock_handler = create_mock_handler()
    records = [{"handler": mock_handler, "age": 30, "gender": "male"}]
    cosinor_age = CosinorAge(records)
    try:
        cosinor_age.plot_predictions()
    except Exception as e:
        assert False, f"plot_predictions raised an exception: {str(e)}"

def test_empty_records():
    try:
        CosinorAge([])
        assert False, "Should raise an exception for empty records"
    except Exception:
        assert True

def test_invalid_record_format():
    try:
        CosinorAge([{"age": 30}])  # Missing handler
        assert False, "Should raise an exception for invalid record format"
    except Exception:
        assert True

def test_multiple_records():
    mock_handler = create_mock_handler()
    records = [
        {"handler": mock_handler, "age": 25, "gender": "female"},
        {"handler": mock_handler, "age": 35, "gender": "male"},
        {"handler": mock_handler, "age": 45, "gender": "unknown"}
    ]
    cosinor_age = CosinorAge(records)
    predictions = cosinor_age.get_predictions()
    
    assert len(predictions) == 3
    for pred in predictions:
        assert 'cosinorage' in pred
        assert pred['cosinorage'] > 0

def test_cosinor_parameters():
    mock_handler = create_mock_handler()
    records = [{"handler": mock_handler, "age": 30, "gender": "male"}]
    cosinor_age = CosinorAge(records)
    predictions = cosinor_age.get_predictions()
    
    # Test if cosinor parameters are within expected ranges
    assert -1 <= predictions[0]['mesor'] <= 1
    assert -1 <= predictions[0]['amp1'] <= 1
    assert -np.pi <= predictions[0]['phi1'] <= np.pi
