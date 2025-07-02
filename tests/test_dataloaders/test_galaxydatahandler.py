import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from cosinorage.datahandlers import GalaxyDataHandler
from cosinorage.datahandlers.utils.frequency_detection import detect_frequency_from_timestamps


class TestGalaxyDataHandler:
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'time': dates,
            'enmo_mg': np.random.randn(100) * 10 + 50
        })
        return data
    
    @pytest.fixture
    def sample_binary_data(self):
        """Create sample binary data structure for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='40ms')  # 25Hz
        data = pd.DataFrame({
            'unix_timestamp_in_ms': dates.astype(np.int64) // 10**6,
            'acceleration_x': np.random.randn(100) * 0.1,
            'acceleration_y': np.random.randn(100) * 0.1,
            'acceleration_z': np.random.randn(100) * 0.1
        })
        return data
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_binary_dir(self, sample_binary_data):
        """Create a temporary directory with binary data files for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample binary files
            for i in range(3):
                file_path = os.path.join(temp_dir, f'data_{i}.bin')
                with open(file_path, 'w') as f:
                    f.write('dummy binary data')
            yield temp_dir

    def test_init_defaults(self):
        """Test GalaxyDataHandler initialization with default parameters"""
        with patch('os.path.isdir', return_value=True):
            with patch.object(GalaxyDataHandler, '_GalaxyDataHandler__load_data'):
                handler = GalaxyDataHandler(galaxy_file_path='/dummy/path')
                
                assert handler.galaxy_file_path == '/dummy/path'
                assert handler.data_format == 'binary'
                assert handler.data_type == 'accelerometer'
                assert handler.time_column == 'unix_timestamp_in_ms'
                assert handler.data_columns == ['acceleration_x', 'acceleration_y', 'acceleration_z']
                assert handler.meta_dict['datasource'] == 'Samsung Galaxy Smartwatch'
                assert handler.meta_dict['data_format'] == 'Binary'
                assert handler.meta_dict['raw_data_type'] == 'Accelerometer'

    def test_init_csv_format(self):
        """Test GalaxyDataHandler initialization with CSV format"""
        with patch('os.path.isfile', return_value=True):
            with patch.object(GalaxyDataHandler, '_GalaxyDataHandler__load_data'):
                handler = GalaxyDataHandler(
                    galaxy_file_path='/dummy/file.csv',
                    data_format='csv'
                )
                
                assert handler.data_format == 'csv'
                assert handler.data_type == 'enmo'
                assert handler.time_column == 'time'
                assert handler.data_columns == ['enmo_mg']
                assert handler.meta_dict['data_format'] == 'CSV'
                assert handler.meta_dict['raw_data_type'] == 'ENMO'

    def test_init_custom_parameters(self):
        """Test GalaxyDataHandler initialization with custom parameters"""
        with patch('os.path.isdir', return_value=True):
            with patch.object(GalaxyDataHandler, '_GalaxyDataHandler__load_data'):
                handler = GalaxyDataHandler(
                    galaxy_file_path='/dummy/path',
                    data_format='binary',
                    data_type='accelerometer',
                    time_column='custom_time',
                    data_columns=['x', 'y', 'z'],
                    preprocess_args={'test': 'value'},
                    verbose=True
                )
                
                assert handler.time_column == 'custom_time'
                assert handler.data_columns == ['x', 'y', 'z']
                assert handler.preprocess_args == {'test': 'value'}

    def test_init_invalid_format(self):
        """Test GalaxyDataHandler initialization with invalid format"""
        with pytest.raises(ValueError, match="data_format must be either 'csv' or 'binary'"):
            GalaxyDataHandler(galaxy_file_path='/dummy/path', data_format='invalid')

    def test_init_invalid_type(self):
        """Test GalaxyDataHandler initialization with invalid data type"""
        with patch('os.path.isdir', return_value=True):
            with pytest.raises(ValueError, match="data_type must be either 'enmo' or 'accelerometer'"):
                GalaxyDataHandler(galaxy_file_path='/dummy/path', data_type='invalid')

    def test_init_csv_with_accelerometer(self):
        """Test that CSV format with accelerometer type raises error"""
        with patch('os.path.isfile', return_value=True):
            with pytest.raises(ValueError, match="CSV format currently only supports 'enmo' data_type"):
                GalaxyDataHandler(
                    galaxy_file_path='/dummy/file.csv',
                    data_format='csv',
                    data_type='accelerometer'
                )

    def test_init_binary_with_enmo(self):
        """Test that binary format with enmo type raises error"""
        with patch('os.path.isdir', return_value=True):
            with pytest.raises(ValueError, match="Binary format currently only supports 'accelerometer' data_type"):
                GalaxyDataHandler(
                    galaxy_file_path='/dummy/path',
                    data_format='binary',
                    data_type='enmo'
                )

    def test_init_invalid_enmo_columns(self):
        """Test that enmo data type with wrong number of columns raises error"""
        with patch('os.path.isfile', return_value=True):
            with pytest.raises(ValueError, match="For 'enmo' data_type, data_columns should contain exactly one column name"):
                GalaxyDataHandler(
                    galaxy_file_path='/dummy/file.csv',
                    data_format='csv',
                    data_columns=['col1', 'col2']
                )

    def test_init_invalid_accelerometer_columns(self):
        """Test that accelerometer data type with wrong number of columns raises error"""
        with patch('os.path.isdir', return_value=True):
            with pytest.raises(ValueError, match="For 'accelerometer' data_type, data_columns should contain exactly three column names"):
                GalaxyDataHandler(
                    galaxy_file_path='/dummy/path',
                    data_columns=['x', 'y']
                )

    def test_init_csv_file_not_found(self):
        """Test that CSV format with non-existent file raises error"""
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(ValueError, match="For CSV format, galaxy_file_path should be a file path"):
                GalaxyDataHandler(
                    galaxy_file_path='/dummy/file.csv',
                    data_format='csv'
                )

    def test_init_binary_dir_not_found(self):
        """Test that binary format with non-existent directory raises error"""
        with patch('os.path.isdir', return_value=False):
            with pytest.raises(ValueError, match="For binary format, galaxy_file_path should be a directory path"):
                GalaxyDataHandler(galaxy_file_path='/dummy/path')

    @patch('cosinorage.datahandlers.utils.galaxy_csv.read_galaxy_csv_data')
    @patch('cosinorage.datahandlers.utils.galaxy_csv.filter_galaxy_csv_data')
    @patch('cosinorage.datahandlers.utils.galaxy_csv.resample_galaxy_csv_data')
    @patch('cosinorage.datahandlers.utils.galaxy_csv.preprocess_galaxy_csv_data')
    @patch('cosinorage.datahandlers.utils.calc_enmo.calculate_minute_level_enmo')
    @patch('pandas.read_csv')
    def test_load_csv_data(self, mock_pd_read_csv, mock_calc_enmo, mock_preprocess, mock_resample, mock_filter, mock_read):
        """Test loading CSV data"""
        with patch('os.path.isfile', return_value=True):
            # Create a DataFrame with at least 4 consecutive days
            timestamps = pd.date_range('2024-01-01', periods=4*1440, freq='1min')
            df = pd.DataFrame({
                'timestamp': timestamps,
                'ENMO': [1, 2, 3, 4] * (1440)
            })
            df = df.set_index('timestamp')
            mock_pd_read_csv.return_value = df.reset_index().copy()
            mock_read.return_value = df.reset_index().copy()
            mock_filter.return_value = df.copy()
            mock_resample.return_value = df.copy()
            mock_preprocess.return_value = df.copy()
            mock_calc_enmo.return_value = df.copy()
            
            handler = GalaxyDataHandler(
                galaxy_file_path='/dummy/file.csv',
                data_format='csv'
            )
            
            # Test that handler was created successfully
            assert handler is not None
            assert handler.data_format == 'csv'
            assert handler.data_type == 'enmo'

    @patch('cosinorage.datahandlers.galaxydatahandler.read_galaxy_binary_data')
    @patch('cosinorage.datahandlers.galaxydatahandler.filter_galaxy_binary_data')
    @patch('cosinorage.datahandlers.galaxydatahandler.resample_galaxy_binary_data')
    @patch('cosinorage.datahandlers.galaxydatahandler.preprocess_galaxy_binary_data')
    @patch('cosinorage.datahandlers.galaxydatahandler.calculate_minute_level_enmo')
    @patch('cosinorage.datahandlers.utils.acceleration_data_to_dataframe')
    @patch('cosinorage.datahandlers.utils.galaxy_binary.load_acceleration_data')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_load_binary_data(self, mock_listdir, mock_isdir, mock_load_accel, mock_accel_to_df, mock_calc_enmo, mock_preprocess, mock_resample, mock_filter, mock_read):
        """Test loading binary data"""
        # Simulate directory structure: /dummy/path/day1/acceleration_data1.binary
        mock_listdir.side_effect = [
            ['day1'],  # First call returns subdirectories
            ['acceleration_data1.binary']  # Second call returns files in subdirectory
        ]
        mock_isdir.side_effect = lambda path: path == '/dummy/path' or path.endswith('day1')
        
        # Mock the acceleration data loader
        mock_accel_data = MagicMock()
        mock_accel_data.data = pd.DataFrame({
            'unix_timestamp_in_ms': pd.date_range('2024-01-01', periods=4*1440, freq='1min').astype(int) // 10**6,
            'acceleration_x': [1, 2, 3, 4] * (1440),
            'acceleration_y': [2, 3, 4, 5] * (1440),
            'acceleration_z': [3, 4, 5, 6] * (1440),
            'sensor_body_location': ['wrist'] * (4*1440),
            'effective_time_frame': ['frame'] * (4*1440)
        })
        mock_load_accel.return_value = mock_accel_data
        
        # Create a DataFrame with the structure that read_galaxy_binary_data expects
        # This should have the original column names that will be renamed
        timestamps = pd.date_range('2024-01-01', periods=4*1440, freq='1min')
        df = pd.DataFrame({
            'unix_timestamp_in_ms': (timestamps.astype(int) // 10**6),
            'acceleration_x': [1, 2, 3, 4] * (1440),
            'acceleration_y': [2, 3, 4, 5] * (1440),
            'acceleration_z': [3, 4, 5, 6] * (1440),
            'sensor_body_location': ['wrist'] * (4*1440),
            'effective_time_frame': ['frame'] * (4*1440)
        })
        
        # Mock the acceleration_data_to_dataframe function
        mock_accel_to_df.return_value = df.copy()
        
        # Mock read_galaxy_binary_data to return the DataFrame after processing
        # This should be the DataFrame after column renaming and processing
        processed_df = df.copy()
        processed_df = processed_df.rename(columns={
            'unix_timestamp_in_ms': 'timestamp',
            'acceleration_x': 'x',
            'acceleration_y': 'y',
            'acceleration_z': 'z'
        })
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], unit='ms')
        processed_df.set_index('timestamp', inplace=True)
        processed_df.drop(columns=['effective_time_frame', 'sensor_body_location'], inplace=True)
        processed_df = processed_df.fillna(0)
        processed_df.sort_index(inplace=True)
        
        mock_read.return_value = processed_df.copy()
        mock_filter.return_value = processed_df.copy()
        mock_resample.return_value = processed_df.copy()
        mock_preprocess.return_value = processed_df.copy()
        mock_calc_enmo.return_value = processed_df.copy()
        
        handler = GalaxyDataHandler(galaxy_file_path='/dummy/path')
        
        # Test that handler was created successfully
        assert handler is not None
        assert handler.data_format == 'binary'
        assert handler.data_type == 'accelerometer'

    def test_unsupported_combination(self):
        """Test that unsupported format-type combinations raise error"""
        with patch('os.path.isdir', return_value=True):
            with patch.object(GalaxyDataHandler, '_GalaxyDataHandler__load_data') as mock_load:
                mock_load.side_effect = ValueError("Unsupported combination")
                
                with pytest.raises(ValueError, match="Unsupported combination"):
                    GalaxyDataHandler(galaxy_file_path='/dummy/path')

    def test_meta_dict_population(self):
        """Test that meta_dict is properly populated"""
        with patch('os.path.isdir', return_value=True):
            with patch.object(GalaxyDataHandler, '_GalaxyDataHandler__load_data'):
                handler = GalaxyDataHandler(
                    galaxy_file_path='/dummy/path',
                    data_format='binary',
                    data_type='accelerometer',
                    time_column='custom_time',
                    data_columns=['x', 'y', 'z']
                )
                
                assert handler.meta_dict['datasource'] == 'Samsung Galaxy Smartwatch'
                assert handler.meta_dict['data_format'] == 'Binary'
                assert handler.meta_dict['raw_data_type'] == 'Accelerometer'
                assert handler.meta_dict['time_column'] == 'custom_time'
                assert handler.meta_dict['data_columns'] == ['x', 'y', 'z'] 

class TestFrequencyDetection:
    """Test cases for frequency detection function."""

    def test_detect_frequency_from_timestamps(self):
        """Test frequency detection from timestamps."""
        # Test with 1-minute frequency
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1min')
        freq = detect_frequency_from_timestamps(timestamps)
        assert freq == 1/60
        
        # Test with 30-second frequency
        timestamps = pd.date_range('2023-01-01', periods=100, freq='30s')
        freq = detect_frequency_from_timestamps(timestamps)
        assert freq == 1/30
        
        # Test with 1-second frequency
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1s')
        freq = detect_frequency_from_timestamps(timestamps)
        assert freq == 1


if __name__ == '__main__':
    pytest.main() 