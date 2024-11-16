import pytest
import pandas as pd
from cosinorage.dataloaders.utils.ukbiobank import read_ukbiobank_data

def test_read_ukbiobank_data(tmp_path):
    # Create test data with valid timestamps and values
    valid_data = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=5, freq='5s').strftime('%Y-%m-%d %H:%M:%S'),
        'ENMO_t': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    valid_file = tmp_path / "valid_data.csv"
    valid_data.to_csv(valid_file, index=False)

    # Test 1: Valid data
    result = read_ukbiobank_data(valid_file, source='uk-biobank')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
    assert 'ENMO' in result.columns
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result['ENMO'].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]

    # Test 2: Invalid source
    with pytest.raises(ValueError, match="Invalid doc_source specified"):
        read_ukbiobank_data(valid_file, source='invalid-source')

    # Test 3: Nonexistent file
    result = read_ukbiobank_data('nonexistent.csv', source='uk-biobank')
    assert isinstance(result, pd.DataFrame)
    assert result.empty

    # Test 4: Inconsistent timestamps
    inconsistent_data = pd.DataFrame({
        'time': [
            '2023-01-01 00:00:00',
            '2023-01-01 00:00:05',
            '2023-01-01 00:00:07'  # Irregular interval
        ],
        'ENMO_t': [0.1, 0.2, 0.3]
    })
    inconsistent_file = tmp_path / "inconsistent_data.csv"
    inconsistent_data.to_csv(inconsistent_file, index=False)
    
    with pytest.raises(ValueError, match="Inconsistent timestamp frequency detected"):
        read_ukbiobank_data(inconsistent_file, source='uk-biobank')

    # Test 5: Invalid timestamp format
    invalid_data = pd.DataFrame({
        'time': ['invalid_date', '2023-01-01', '2023-01-02'],
        'ENMO_t': [0.1, 0.2, 0.3]
    })
    invalid_file = tmp_path / "invalid_data.csv"
    invalid_data.to_csv(invalid_file, index=False)
    
    result = read_ukbiobank_data(invalid_file, source='uk-biobank')
    assert isinstance(result, pd.DataFrame)
    assert result.empty