from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cosinorage.datahandlers.utils.generic import read_generic_xD_data


class TestGenericUtils:

    @pytest.fixture
    def sample_csv_file_1d(self, tmp_path):
        """Create a sample 1D CSV file for testing"""
        file_path = tmp_path / "test_1d.csv"

        # Create sample data with 1-minute intervals
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(100)]
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "counts": np.random.randint(0, 1000, 100),
            }
        )
        data.to_csv(file_path, index=False)
        return str(file_path)

    @pytest.fixture
    def sample_csv_file_3d(self, tmp_path):
        """Create a sample 3D CSV file for testing"""
        file_path = tmp_path / "test_3d.csv"

        # Create sample data with 1-minute intervals
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(100)]
        data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "x": np.random.rand(100) * 2 - 1,
                "y": np.random.rand(100) * 2 - 1,
                "z": np.random.rand(100) * 2 - 1,
            }
        )
        data.to_csv(file_path, index=False)
        return str(file_path)

    def test_read_generic_xD_1d_default_columns(self, sample_csv_file_1d):
        """Test read_generic_xD with 1D data and default column names"""
        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=sample_csv_file_1d,
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
            )

            # Check that the result is a DataFrame with correct structure
            assert isinstance(result, pd.DataFrame)
            assert "ENMO" in result.columns
            assert result.index.name == "timestamp"
            assert len(result) == 100

            # Check metadata
            assert meta_dict["raw_n_datapoints"] == 100
            assert meta_dict["raw_data_unit"] == "counts"
            assert meta_dict["raw_data_frequency"] == "1.0Hz"
            assert meta_dict["sf"] == 1.0

    def test_read_generic_xD_3d_default_columns(self, sample_csv_file_3d):
        """Test read_generic_xD with 3D data and default column names"""
        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 25.0

            result = read_generic_xD_data(
                file_path=sample_csv_file_3d,
                data_type="accelerometer",
                meta_dict=meta_dict,
                n_dimensions=3,
            )

            # Check that the result is a DataFrame with correct structure
            assert isinstance(result, pd.DataFrame)
            assert "x" in result.columns
            assert "y" in result.columns
            assert "z" in result.columns
            assert result.index.name == "timestamp"
            assert len(result) == 100

            # Check metadata
            assert meta_dict["raw_n_datapoints"] == 100
            assert meta_dict["raw_data_unit"] == "mg"
            assert meta_dict["raw_data_frequency"] == "25.0Hz"
            assert meta_dict["sf"] == 25.0

    def test_read_generic_xD_custom_columns(self, sample_csv_file_1d):
        """Test read_generic_xD with custom column names"""
        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=sample_csv_file_1d,
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
                time_column="timestamp",
                data_columns=["counts"],
            )

            # Check that the result is a DataFrame with correct structure
            assert isinstance(result, pd.DataFrame)
            assert "ENMO" in result.columns
            assert result.index.name == "timestamp"
            assert len(result) == 100

    def test_read_generic_xD_custom_time_column(self, sample_csv_file_1d):
        """Test read_generic_xD with custom time column name"""
        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=sample_csv_file_1d,
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
                time_column="timestamp",
            )

            # Check that the result is a DataFrame with correct structure
            assert isinstance(result, pd.DataFrame)
            assert "ENMO" in result.columns
            assert result.index.name == "timestamp"
            assert len(result) == 100

    def test_read_generic_xD_verbose_output(self, sample_csv_file_1d, capsys):
        """Test read_generic_xD with verbose output"""
        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=sample_csv_file_1d,
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
                verbose=True,
            )

            # Check that verbose output was printed
            captured = capsys.readouterr()
            assert "Read csv file from" in captured.out
            assert "Loaded 100 Count data records" in captured.out

    def test_read_generic_xD_invalid_n_dimensions(self):
        """Test read_generic_xD with invalid n_dimensions"""
        meta_dict = {}

        with pytest.raises(
            ValueError, match="n_dimensions must be either 1 or 3"
        ):
            read_generic_xD_data(
                file_path="/dummy/path.csv",
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=2,
            )

    def test_read_generic_xD_mismatched_columns(self):
        """Test read_generic_xD with mismatched data_columns and n_dimensions"""
        meta_dict = {}

        with pytest.raises(
            ValueError,
            match="n_dimensions must be equal to the number of data columns",
        ):
            read_generic_xD_data(
                file_path="/dummy/path.csv",
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
                data_columns=["x", "y", "z"],  # 3 columns for 1 dimension
            )

    def test_read_generic_xD_handles_missing_values(self, tmp_path):
        """Test read_generic_xD handles missing values correctly"""
        file_path = tmp_path / "test_missing.csv"

        # Create data with missing values
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="1min"
                ),
                "counts": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            }
        )
        data.to_csv(file_path, index=False)

        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=str(file_path),
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
            )

            # Check that missing values were filled with 0
            assert not result["ENMO"].isna().any()
            assert (result["ENMO"] == 0).sum() == 2

    def test_read_generic_xD_timestamp_processing(self, tmp_path):
        """Test read_generic_xD correctly processes timestamps"""
        file_path = tmp_path / "test_timestamps.csv"

        # Create data with timezone-aware timestamps
        timestamps = pd.date_range(
            "2024-01-01", periods=10, freq="1min", tz="UTC"
        )
        data = pd.DataFrame(
            {"timestamp": timestamps, "counts": np.random.randint(0, 1000, 10)}
        )
        data.to_csv(file_path, index=False)

        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=str(file_path),
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
            )

            # Check that timestamps are timezone-naive
            if isinstance(result.index, pd.DatetimeIndex):
                assert result.index.tz is None
            # Check that timestamps are sorted
            assert result.index.is_monotonic_increasing

    def test_read_generic_xD_metadata_population(self, sample_csv_file_1d):
        """Test read_generic_xD properly populates metadata"""
        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 25.0

            result = read_generic_xD_data(
                file_path=sample_csv_file_1d,
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
            )

            # Check all metadata fields
            assert "raw_n_datapoints" in meta_dict
            assert "raw_start_datetime" in meta_dict
            assert "raw_end_datetime" in meta_dict
            assert "sf" in meta_dict
            assert "raw_data_frequency" in meta_dict
            assert "raw_data_unit" in meta_dict

            assert meta_dict["raw_n_datapoints"] == 100
            assert meta_dict["raw_data_unit"] == "counts"
            assert meta_dict["raw_data_frequency"] == "25.0Hz"
            assert meta_dict["sf"] == 25.0

    def test_read_generic_xD_column_mapping_1d(self, tmp_path):
        """Test read_generic_xD correctly maps columns for 1D data"""
        file_path = tmp_path / "test_mapping_1d.csv"

        data = pd.DataFrame(
            {
                "custom_time": pd.date_range(
                    "2024-01-01", periods=10, freq="1min"
                ),
                "custom_data": np.random.randint(0, 1000, 10),
            }
        )
        data.to_csv(file_path, index=False)

        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=str(file_path),
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
                time_column="custom_time",
                data_columns=["custom_data"],
            )

            # Check that columns were correctly mapped
            assert "ENMO" in result.columns
            assert result.index.name == "timestamp"
            assert len(result) == 10

    def test_read_generic_xD_column_mapping_3d(self, tmp_path):
        """Test read_generic_xD correctly maps columns for 3D data"""
        file_path = tmp_path / "test_mapping_3d.csv"

        data = pd.DataFrame(
            {
                "custom_time": pd.date_range(
                    "2024-01-01", periods=10, freq="1min"
                ),
                "accel_x": np.random.rand(10),
                "accel_y": np.random.rand(10),
                "accel_z": np.random.rand(10),
            }
        )
        data.to_csv(file_path, index=False)

        meta_dict = {}

        with patch(
            "cosinorage.datahandlers.utils.generic.detect_frequency_from_timestamps"
        ) as mock_detect:
            mock_detect.return_value = 1.0

            result = read_generic_xD_data(
                file_path=str(file_path),
                data_type="accelerometer",
                meta_dict=meta_dict,
                n_dimensions=3,
                time_column="custom_time",
                data_columns=["accel_x", "accel_y", "accel_z"],
            )

            # Check that columns were correctly mapped
            assert "x" in result.columns
            assert "y" in result.columns
            assert "z" in result.columns
            assert result.index.name == "timestamp"
            assert len(result) == 10

    def test_read_generic_xD_file_not_found(self):
        """Test read_generic_xD raises error for non-existent file"""
        meta_dict = {}

        with pytest.raises(FileNotFoundError):
            read_generic_xD_data(
                file_path="/non/existent/file.csv",
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
            )

    def test_read_generic_xD_invalid_csv_format(self, tmp_path):
        """Test read_generic_xD handles invalid CSV format gracefully"""
        file_path = tmp_path / "test_invalid.csv"

        # Create a file that's not a valid CSV
        with open(file_path, "w") as f:
            f.write("This is not a CSV file\n")
            f.write("It has no proper structure\n")

        meta_dict = {}

        with pytest.raises(
            Exception
        ):  # Should raise some kind of parsing error
            read_generic_xD_data(
                file_path=str(file_path),
                data_type="alternative_count",
                meta_dict=meta_dict,
                n_dimensions=1,
            )
