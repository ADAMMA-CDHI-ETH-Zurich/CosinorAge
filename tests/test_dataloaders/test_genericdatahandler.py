import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os

from cosinorage.datahandlers.genericdatahandler import GenericDataHandler


class TestGenericDataHandler:
    """Test cases for GenericDataHandler class."""

    def test_init_enmo_data_type(self):
        """Test initialization with ENMO data type."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_csv_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_csv_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_csv_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='enmo',
                                data_columns=['enmo']
                            )
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()
                            mock_calc.assert_called_once()
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_init_accelerometer_data_type(self):
        """Test initialization with accelerometer data type."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_binary_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_binary_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_binary_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='accelerometer',
                                data_columns=['x', 'y', 'z']
                            )
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()
                            mock_calc.assert_called_once()
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_init_alternative_count_data_type(self):
        """Test initialization with alternative count data type."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_csv_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_csv_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_csv_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='alternative_count',
                                data_columns=['counts']
                            )
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()
                            mock_calc.assert_called_once()
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_csv_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_csv_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_csv_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/custom/path.csv',
                                data_format='csv',
                                data_type='enmo',
                                time_column='custom_time',
                                data_columns=['custom_enmo'],
                                preprocess_args={'custom_param': 'value'}
                            )
                            assert handler.file_path == '/custom/path.csv'
                            assert handler.data_format == 'csv'
                            assert handler.data_type == 'enmo'
                            assert handler.time_column == 'custom_time'
                            assert handler.data_columns == ['custom_enmo']
                            assert handler.preprocess_args == {'custom_param': 'value'}

    def test_init_invalid_data_format(self):
        """Test initialization with invalid data format."""
        with pytest.raises(ValueError, match="Data format must be either 'csv'"):
            GenericDataHandler(
                file_path='/dummy/path.csv',
                data_format='invalid',
                data_type='enmo'
            )

    def test_init_invalid_data_type(self):
        """Test initialization with invalid data type."""
        with pytest.raises(ValueError, match="Data type must be either 'enmo', 'accelerometer' or 'alternative_count'"):
            GenericDataHandler(
                file_path='/dummy/path.csv',
                data_format='csv',
                data_type='invalid'
            )

    def test_init_data_columns_mismatch(self):
        """Test initialization with mismatched data columns."""
        with pytest.raises(ValueError, match="n_dimensions must be equal to the number of data columns"):
            GenericDataHandler(
                file_path='/dummy/path.csv',
                data_format='csv',
                data_type='accelerometer',
                data_columns=['x', 'y']  # Missing z
            )

    def test_load_data_enmo_flow(self):
        """Test the complete data loading flow for ENMO data."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_csv_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_csv_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_csv_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='enmo',
                                data_columns=['enmo']
                            )
                            assert mock_read.called
                            assert mock_filter.called
                            assert mock_resample.called
                            assert mock_preprocess.called
                            assert mock_calc.called
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_load_data_accelerometer_flow(self):
        """Test the complete data loading flow for accelerometer data."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_binary_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_binary_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_binary_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='accelerometer',
                                data_columns=['x', 'y', 'z']
                            )
                            assert mock_read.called
                            assert mock_filter.called
                            assert mock_resample.called
                            assert mock_preprocess.called
                            assert mock_calc.called
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_inheritance_from_datahandler(self):
        """Test that GenericDataHandler properly inherits from DataHandler."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_csv_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_csv_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_csv_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            mock_read.return_value = mock_df
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='enmo',
                                data_columns=['enmo']
                            )
                            assert hasattr(handler, 'get_raw_data')
                            assert hasattr(handler, 'get_sf_data')
                            assert hasattr(handler, 'get_ml_data')
                            assert hasattr(handler, 'get_meta_data')
                            raw_data = handler.get_raw_data()
                            sf_data = handler.get_sf_data()
                            ml_data = handler.get_ml_data()
                            meta_data = handler.get_meta_data()
                            assert raw_data is not None
                            assert sf_data is not None
                            assert ml_data is not None
                            assert meta_data is not None

    def test_metadata_population(self):
        """Test that metadata is properly populated during data loading."""
        with patch('cosinorage.datahandlers.genericdatahandler.read_generic_xD') as mock_read:
            with patch('cosinorage.datahandlers.genericdatahandler.filter_galaxy_csv_data', side_effect=lambda df, *a, **kw: df) as mock_filter:
                with patch('cosinorage.datahandlers.genericdatahandler.resample_galaxy_csv_data') as mock_resample:
                    with patch('cosinorage.datahandlers.genericdatahandler.preprocess_galaxy_csv_data') as mock_preprocess:
                        with patch('cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo') as mock_calc:
                            mock_df = pd.DataFrame({
                                'TIMESTAMP': pd.date_range('2024-01-01', periods=4*1440, freq='1min'),
                                'ENMO': np.random.rand(4*1440)
                            }).set_index('TIMESTAMP')
                            
                            def mock_read_with_metadata(file_path, meta_dict, *args, **kwargs):
                                meta_dict['raw_n_datapoints'] = mock_df.shape[0]
                                meta_dict['raw_start_datetime'] = mock_df.index.min()
                                meta_dict['raw_end_datetime'] = mock_df.index.max()
                                meta_dict['sf'] = 1
                                meta_dict['raw_data_frequency'] = '1Hz'
                                meta_dict['raw_data_type'] = 'Counts'
                                meta_dict['raw_data_unit'] = 'counts'
                                return mock_df
                            
                            mock_read.side_effect = mock_read_with_metadata
                            mock_resample.return_value = mock_df
                            mock_preprocess.return_value = mock_df
                            mock_calc.return_value = mock_df
                            handler = GenericDataHandler(
                                file_path='/dummy/path.csv',
                                data_format='csv',
                                data_type='enmo',
                                time_column='custom_time',
                                data_columns=['custom_enmo']
                            )
                            meta_data = handler.get_meta_data()
                            assert 'raw_n_datapoints' in meta_data
                            assert 'raw_start_datetime' in meta_data
                            assert 'raw_end_datetime' in meta_data
                            assert 'raw_data_frequency' in meta_data
                            assert 'raw_data_type' in meta_data
                            assert 'raw_data_unit' in meta_data 