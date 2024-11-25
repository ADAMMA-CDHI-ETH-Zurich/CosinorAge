import pandas as pd
from typing import Union, Any

def read_ukbiobank_data(file_path: str, meta_dict: dict = {}) -> Union[pd.DataFrame, tuple[Any, Union[float, Any]]]:
    """
    Read UK Biobank data from a CSV file and process it.

    Args:
        file_path (str): The path to the CSV file containing the data.
        source (str): The source of the data, should be 'uk-biobank'.

    Returns:
        Union[pd.DataFrame, tuple[Any, Union[float, Any]]]: A DataFrame containing the processed data,
        or an empty DataFrame in case of an error.
    """

    # Read the CSV file
    try:
        data = pd.read_csv(file_path)
        data = data.sort_values(by=time_col)
        data.rename(columns={enmo_col: 'ENMO'}, inplace=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

    # Convert timestamps to datetime format
    try:
        data[time_col] = pd.to_datetime(data[time_col], format='mixed')
        data.rename(columns={time_col: 'TIMESTAMP'}, inplace=True)
    except Exception as e:
        print(f"Error converting timestamps: {e}")
        return pd.DataFrame()

    # check if timestamp frequency is consistent up to 1ms
    time_diffs = data['TIMESTAMP'].diff().dropna()
    unique_diffs = time_diffs.unique()
    if not len(unique_diffs) == 1:
        raise ValueError("Inconsistent timestamp frequency detected.")

    data.set_index('TIMESTAMP', inplace=True)

    return data[['ENMO']]
