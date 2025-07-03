from unittest.mock import patch

import pandas as pd
import pytest

from cosinorage.datahandlers.genericdatahandler import GenericDataHandler


class TestGenericDataHandler:
    """Test cases for GenericDataHandler class."""

    def test_init_enmo_data_type(self):
        """Test initialization with ENMO data type."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv", data_type="enmo"
                            )

                            assert handler.data_type == "enmo-mg"
                            assert handler.data_columns == ["enmo"]
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()

    def test_init_accelerometer_data_type(self):
        """Test initialization with accelerometer data type."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {
                                    "x": [1, 2, 3],
                                    "y": [1, 2, 3],
                                    "z": [1, 2, 3],
                                },
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {
                                    "x": [1, 2, 3],
                                    "y": [1, 2, 3],
                                    "z": [1, 2, 3],
                                },
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {
                                    "x": [1, 2, 3],
                                    "y": [1, 2, 3],
                                    "z": [1, 2, 3],
                                },
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv",
                                data_type="accelerometer",
                            )

                            assert handler.data_type == "accelerometer-mg"
                            assert handler.data_columns == ["x", "y", "z"]
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()

    def test_init_alternative_count_data_type(self):
        """Test initialization with alternative count data type."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {"counts": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {"counts": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {"counts": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv",
                                data_type="alternative_count",
                            )

                            assert handler.data_type == "alternative_count"
                            assert handler.data_columns == ["counts"]
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv",
                                data_type="enmo",
                                time_column="custom_time",
                                data_columns=["custom_enmo"],
                                preprocess_args={"test_param": "test_value"},
                                verbose=True,
                            )

                            assert handler.time_column == "custom_time"
                            assert handler.data_columns == ["custom_enmo"]
                            assert handler.preprocess_args == {
                                "test_param": "test_value"
                            }
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()

    def test_init_invalid_data_format(self):
        """Test initialization with invalid data format."""
        with pytest.raises(
            ValueError, match="Data format must be either 'csv'"
        ):
            GenericDataHandler(
                file_path="/dummy/path.csv",
                data_format="invalid",
                data_type="enmo",
            )

    def test_init_invalid_data_type(self):
        """Test initialization with invalid data type."""
        with pytest.raises(
            ValueError,
            match="Data type must be either 'enmo-mg', 'enmo-g', 'accelerometer-mg', 'accelerometer-g', 'accelerometer-ms2' or 'alternative_count'",
        ):
            GenericDataHandler(
                file_path="/dummy/path.csv",
                data_format="csv",
                data_type="invalid",
            )

    def test_init_data_columns_mismatch(self):
        """Test initialization with mismatched data columns."""
        with pytest.raises(
            ValueError,
            match="n_dimensions must be equal to the number of data columns",
        ):
            GenericDataHandler(
                file_path="/dummy/path.csv",
                data_format="csv",
                data_type="accelerometer",
                data_columns=["x", "y"],  # Missing z
            )

    def test_load_data_enmo_flow(self):
        """Test the complete data loading flow for ENMO data."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv", data_type="enmo"
                            )

                            # Check that all processing steps were called
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()

                            # Check that data was properly set
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_load_data_accelerometer_flow(self):
        """Test the complete data loading flow for accelerometer data."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {
                                    "x": [1, 2, 3],
                                    "y": [1, 2, 3],
                                    "z": [1, 2, 3],
                                },
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {
                                    "x": [1, 2, 3],
                                    "y": [1, 2, 3],
                                    "z": [1, 2, 3],
                                },
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {
                                    "x": [1, 2, 3],
                                    "y": [1, 2, 3],
                                    "z": [1, 2, 3],
                                },
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv",
                                data_type="accelerometer",
                            )

                            # Check that all processing steps were called
                            mock_read.assert_called_once()
                            mock_filter.assert_called_once()
                            mock_resample.assert_called_once()
                            mock_preprocess.assert_called_once()

                            # Check that data was properly set
                            assert handler.raw_data is not None
                            assert handler.sf_data is not None
                            assert handler.ml_data is not None

    def test_inheritance_from_datahandler(self):
        """Test that GenericDataHandler properly inherits from DataHandler."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv", data_type="enmo"
                            )

                            # Check inheritance
                            from cosinorage.datahandlers.datahandler import \
                                DataHandler

                            assert isinstance(handler, DataHandler)

                            # Check that DataHandler methods are available
                            assert hasattr(handler, "get_raw_data")
                            assert hasattr(handler, "get_ml_data")

    def test_metadata_population(self):
        """Test that metadata is properly populated during data loading."""
        with patch(
            "cosinorage.datahandlers.genericdatahandler.read_generic_xD_data"
        ) as mock_read:
            with patch(
                "cosinorage.datahandlers.genericdatahandler.filter_generic_data",
                side_effect=lambda df, *a, **kw: df,
            ) as mock_filter:
                with patch(
                    "cosinorage.datahandlers.genericdatahandler.resample_generic_data"
                ) as mock_resample:
                    with patch(
                        "cosinorage.datahandlers.genericdatahandler.preprocess_generic_data"
                    ) as mock_preprocess:
                        with patch(
                            "cosinorage.datahandlers.genericdatahandler.calculate_minute_level_enmo"
                        ) as mock_enmo:
                            mock_read.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_resample.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_preprocess.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )
                            mock_enmo.return_value = pd.DataFrame(
                                {"enmo": [1, 2, 3]},
                                index=pd.date_range(
                                    "2024-01-01", periods=3, freq="1min"
                                ),
                            )

                            handler = GenericDataHandler(
                                file_path="/dummy/path.csv",
                                data_type="enmo",
                                time_column="custom_time",
                                data_columns=["custom_enmo"],
                            )

                            # Check metadata
                            assert handler.meta_dict["datasource"] == "Generic"
                            assert handler.meta_dict["data_format"] == "CSV"
                            assert handler.meta_dict["raw_data_type"] == "ENMO"
                            assert (
                                handler.meta_dict["time_column"]
                                == "custom_time"
                            )
                            assert handler.meta_dict["data_columns"] == [
                                "custom_enmo"
                            ]
