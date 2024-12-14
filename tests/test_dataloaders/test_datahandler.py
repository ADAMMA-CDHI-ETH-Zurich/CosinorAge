import pytest
import pandas as pd
from cosinorage.datahandlers.datahandler import DataHandler

class TestDataHandler:
    @pytest.fixture
    def data_handler(self):
        """Create a basic DataHandler instance for testing"""
        return DataHandler()

    def test_init(self, data_handler):
        """Test initialization of DataHandler"""
        assert data_handler.raw_data is None
        assert data_handler.sf_data is None
        assert data_handler.ml_data is None
        assert isinstance(data_handler.meta_dict, dict)
        assert len(data_handler.meta_dict) == 0

    def test_load_data_not_implemented(self, data_handler):
        """Test that __load_data raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            data_handler._DataHandler__load_data()

    def test_save_data_without_loading(self, data_handler, tmp_path):
        """Test save_data raises error when data not loaded"""
        output_path = tmp_path / "test_output.csv"
        with pytest.raises(ValueError, match="Data has not been loaded"):
            data_handler.save_data(str(output_path))

    def test_save_data_with_data(self, data_handler, tmp_path):
        """Test save_data successfully saves data"""
        # Create sample data
        test_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        data_handler.ml_data = test_data
        
        output_path = tmp_path / "test_output.csv"
        data_handler.save_data(str(output_path))
        
        # Verify file was created and contains correct data
        assert output_path.exists()
        saved_data = pd.read_csv(str(output_path))
        pd.testing.assert_frame_equal(saved_data, test_data)

    def test_get_raw_data(self, data_handler):
        """Test get_raw_data returns None when no data loaded"""
        assert data_handler.get_raw_data() is None

    def test_get_sf_data_without_loading(self, data_handler):
        """Test get_sf_data raises error when data not loaded"""
        with pytest.raises(ValueError, match="Data has not been loaded"):
            data_handler.get_sf_data()

    def test_get_ml_data_without_loading(self, data_handler):
        """Test get_ml_data raises error when data not loaded"""
        with pytest.raises(ValueError, match="Data has not been loaded"):
            data_handler.get_ml_data()

    def test_get_meta_data(self, data_handler):
        """Test get_meta_data returns empty dict initially"""
        assert data_handler.get_meta_data() == {}

    def test_get_data_with_loaded_data(self, data_handler):
        """Test getter methods with sample data"""
        # Create sample data
        raw_data = pd.DataFrame({'raw': [1, 2, 3]})
        sf_data = pd.DataFrame({'sf': [4, 5, 6]})
        ml_data = pd.DataFrame({'ml': [7, 8, 9]})
        meta_dict = {'key': 'value'}

        # Set data
        data_handler.raw_data = raw_data
        data_handler.sf_data = sf_data
        data_handler.ml_data = ml_data
        data_handler.meta_dict = meta_dict

        # Test getters
        pd.testing.assert_frame_equal(data_handler.get_raw_data(), raw_data)
        pd.testing.assert_frame_equal(data_handler.get_sf_data(), sf_data)
        pd.testing.assert_frame_equal(data_handler.get_ml_data(), ml_data)
        assert data_handler.get_meta_data() == meta_dict
