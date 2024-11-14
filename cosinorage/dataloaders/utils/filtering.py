import pandas as pd


def filter_incomplete_days(df: pd.DataFrame, data_freq: float) -> pd.DataFrame:
    """
    Filter out data from incomplete days to ensure 24-hour data periods.

    This function removes data from the first and last days in the DataFrame
    to ensure that only complete 24-hour data is retained.

    Args:
        data_all (pd.DataFrame): DataFrame with a 'TIMESTAMP' column in datetime
            format, which is used to determine the day.

    Returns:
        pd.DataFrame: Filtered DataFrame excluding the first and last days.
        If there
        are fewer than two unique dates in the data, an empty DataFrame is
        returned.
    """

    # Filter out incomplete days
    try:
        # Calculate expected number of data points for a full 24-hour day
        expected_points_per_day = data_freq * 60 * 60 * 24

        # Extract the date from each timestamp
        _df = df.copy()
        # timestamp is index
        _df['DATE'] = _df.index.date

        # Count data points for each day
        daily_counts = _df.groupby('DATE').size()

        # Identify complete days based on expected number of data points
        complete_days = daily_counts[
            daily_counts >= expected_points_per_day].index

        # Filter the DataFrame to include only rows from complete days
        filtered_df = _df[_df['DATE'].isin(complete_days)]

        # Drop the helper 'DATE' column before returning
        return filtered_df.drop(columns=['DATE'])

    except Exception as e:
        print(f"Error filtering incomplete days: {e}")
        return pd.DataFrame()
