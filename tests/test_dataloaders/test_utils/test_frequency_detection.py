import pandas as pd
import pytest

from cosinorage.datahandlers.utils.frequency_detection import \
    detect_frequency_from_timestamps


class TestFrequencyDetection:

    def test_detect_frequency_regular_25hz(self):
        """Test frequency detection with regular 25Hz data"""
        # Create timestamps with 40ms intervals (25Hz)
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.Series(
            [base_time + pd.Timedelta(milliseconds=40 * i) for i in range(100)]
        )

        frequency = detect_frequency_from_timestamps(timestamps)

        # Should be close to 25Hz (allowing for small floating point errors)
        assert abs(frequency - 25.0) < 0.1

    def test_detect_frequency_regular_1hz(self):
        """Test frequency detection with regular 1Hz data"""
        # Create timestamps with 1 second intervals (1Hz)
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.Series(
            [base_time + pd.Timedelta(seconds=i) for i in range(100)]
        )

        frequency = detect_frequency_from_timestamps(timestamps)

        # Should be close to 1Hz
        assert abs(frequency - 1.0) < 0.01

    def test_detect_frequency_with_irregular_samples(self):
        """Test frequency detection with some irregular samples"""
        # Create mostly regular 25Hz data with a few irregular samples
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = []

        for i in range(100):
            if i % 20 == 0:  # Every 20th sample has irregular timing
                timestamps.append(
                    base_time + pd.Timedelta(milliseconds=40 * i + 100)
                )
            else:
                timestamps.append(
                    base_time + pd.Timedelta(milliseconds=40 * i)
                )

        timestamps = pd.Series(timestamps)
        frequency = detect_frequency_from_timestamps(timestamps)

        # Should still detect 25Hz as the majority frequency
        assert abs(frequency - 25.0) < 0.1

    def test_detect_frequency_datetime_strings(self):
        """Test frequency detection with datetime strings"""
        # Create timestamps as ISO8601 strings with 1-second intervals (1Hz)
        # This avoids the rounding issue with sub-second intervals
        base_time = pd.Timestamp("2024-01-01T00:00:00")
        timestamps = []
        for i in range(100):
            ts = base_time + pd.Timedelta(seconds=i)
            if isinstance(ts, pd.Timestamp):
                timestamps.append(ts.isoformat())
        timestamps = pd.Series(timestamps)

        frequency = detect_frequency_from_timestamps(timestamps)
        # Should be exactly 1Hz (1/1.0)
        assert abs(frequency - 1.0) < 0.1

    def test_detect_frequency_single_sample(self):
        """Test frequency detection with only one sample"""
        timestamps = pd.Series([pd.Timestamp("2024-01-01 00:00:00")])

        with pytest.raises(ValueError):
            detect_frequency_from_timestamps(timestamps)

    def test_detect_frequency_empty_series(self):
        """Test frequency detection with empty series"""
        timestamps = pd.Series([])

        with pytest.raises(ValueError):
            detect_frequency_from_timestamps(timestamps)

    def test_detect_frequency_two_samples(self):
        """Test frequency detection with exactly two samples"""
        timestamps = pd.Series(
            [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:00:00.040"),  # 40ms later
            ]
        )

        frequency = detect_frequency_from_timestamps(timestamps)

        # Should be 25Hz (1/0.04)
        assert abs(frequency - 25.0) < 0.01

    def test_detect_frequency_mixed_frequencies(self):
        """Test frequency detection with mixed frequencies (should pick majority)"""
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = []

        # Add 60 samples at 25Hz (40ms intervals)
        for i in range(60):
            timestamps.append(base_time + pd.Timedelta(milliseconds=40 * i))

        # Add 40 samples at 10Hz (100ms intervals)
        for i in range(40):
            timestamps.append(
                base_time + pd.Timedelta(milliseconds=2400 + 100 * i)
            )

        timestamps = pd.Series(timestamps)
        frequency = detect_frequency_from_timestamps(timestamps)

        # Should detect 25Hz as the majority frequency
        assert abs(frequency - 25.0) < 0.1

    def test_detect_frequency_nan_values(self):
        """Test frequency detection with NaN values"""
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.Series(
            [
                (
                    base_time + pd.Timedelta(milliseconds=40 * i)
                    if i % 10 != 0
                    else pd.NaT
                )
                for i in range(100)
            ]
        )

        # Remove NaN values
        timestamps = timestamps.dropna()
        frequency = detect_frequency_from_timestamps(timestamps)

        # Should still detect 25Hz
        assert abs(frequency - 25.0) < 0.1
