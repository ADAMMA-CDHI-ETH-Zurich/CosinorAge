import pandas as pd
from typing import Union, Any
import os
import glob
import pyreadr

def read_ukbiobank_data(qc_file_path: str, enmo_file_dir: str, person_id: str, meta_dict: dict = {}) -> Union[pd.DataFrame, tuple[Any, Union[float, Any]]]:
    """
    Read UK Biobank data from a CSV file and process it.

    Args:
        file_path (str): The path to the CSV file containing the data.
        source (str): The source of the data, should be 'uk-biobank'.

    Returns:
        Union[pd.DataFrame, tuple[Any, Union[float, Any]]]: A DataFrame containing the processed data,
        or an empty DataFrame in case of an error.
    """
    # check if qa_file_path and acc_file_path exist
    if not os.path.exists(qc_file_path):
        raise FileNotFoundError(f"QA file does not exist: {qc_file_path}")
    if not os.path.exists(enmo_file_dir):
        raise FileNotFoundError(f"ENMO file directory does not exist: {enmo_file_dir}")

    qa_data = pd.read_csv(qc_file_path)

    if person_id not in qa_data['eid'].values:
        raise ValueError(f"Person ID {person_id} not found in QA file")

    acc_qc = qa_data[qa_data["eid"] == person_id]

    #Exclude participants with data problems - filter rows where `acc_data_problem` is blank
    acc_qc = qa_data[qa_data["acc_data_problem"].isnull() | (qa_data["acc_data_problem"] == "")]

    if acc_qc.empty:
        raise ValueError(f"Person ID {person_id} has no valid accelerometer data - check for data problems")

    # Exclude participants with poor wear time - filter rows where `acc_weartime` is "Yes"
    acc_qc = acc_qc[acc_qc["acc_weartime"] == "Yes"]

    if acc_qc.empty:
        raise ValueError(f"Person ID {person_id} has no valid accelerometer data - check for wear time")
    # Exclude participants with poor calibration - filter rows where `acc_calibration` is "Yes"
    acc_qc = acc_qc[acc_qc["acc_calibration"] == "Yes"]

    if acc_qc.empty:
        raise ValueError(f"Person ID {person_id} has no valid accelerometer data - check for calibration")

    # Exclude participants not calibrated on their own data - filter rows where `acc_owndata` is "Yes"
    acc_qc = acc_qc[acc_qc["acc_owndata"] == "Yes"]

    if acc_qc.empty:
        raise ValueError(f"Person ID {person_id} has no valid accelerometer data - check for own data calibration")

    # Exclude participants with interrupted recording periods - filter rows where `acc_interrupt_period` is 0
    acc_qc = acc_qc[acc_qc["acc_interrupt_period"] == 0]

    if acc_qc.empty:
        raise ValueError(f"Person ID {person_id} has no valid accelerometer data - check for interrupted recording periods")

    # read acc file
    enmo_file_names = glob.glob(os.path.join(enmo_file_dir, "ukb_enmo_*.RDS"))

    keeplist = []

    for file in enmo_file_names:
        result = pyreadr.read_r(file)
        enmo_data = result[None]

        # Filter for person_id
        enmo_data = enmo_data[enmo_data['eid'] == person_id]

        # Filter out day 1 and calculate 5-minute epochs
        enmo_data = enmo_data[enmo_data['day'] >= 2]
        enmo_data['myepoch'] = (12 * enmo_data['hour'].astype(int) + (enmo_data['minute'] // 5 + 1).astype(int))

        # Calculate how many epochs per day
        check = (enmo_data.groupby(['eid', 'day'])['myepoch']
                .nunique()
             .reset_index(name='n_epoch'))

        # Keep only days with 288 epochs
        ukb_keep = check[check['n_epoch'] == 288][['eid', 'day']]
        keeplist.append(ukb_keep)

    # Concatenate all kept data
    data = pd.concat(keeplist, ignore_index=True)



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
    """

    return data[['ENMO']]
