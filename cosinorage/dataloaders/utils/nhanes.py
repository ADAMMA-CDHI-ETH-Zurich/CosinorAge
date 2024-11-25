import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

from cosinorage.dataloaders.utils.calc_enmo import calculate_enmo

def read_nhanes_data(file_dir: str, meta_dict: dict = {}, verbose: bool = False, person_id: str = None) -> pd.DataFrame:
    # list all files in directory starting with PAX
    pax_files = [f for f in os.listdir(file_dir) if f.startswith('PAX')]
    # for each file starting with PAXDAY check if PAXHD and PAXMIN are present
    versions = []
    for file in pax_files:
        if file.startswith('PAXDAY'):
            version = file.split("_")[1]
            if f'PAXHD_{version}.xpt' in pax_files and f'PAXMIN_{version}.xpt' in pax_files:
                versions.append(version)

    if verbose:
        print(f"Found {len(versions)} versions of NHANES data")

    # read all day-level files
    day_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading day-level files"):
        curr = pd.read_sas(f"{file_dir}/PAXDAY_{version}.xpt")
        curr = curr[curr['seqn'] == person_id]
        day_x = pd.concat([day_x, curr], ignore_index=True)

    day_x.columns = day_x.columns.str.lower()
    day_x = remove_bytes(day_x)

    if verbose:
        print(f"Read {day_x.shape[0]} day-level records for person {person_id}")

    # check data quality flags
    day_x = day_x[day_x['paxqfd'] < 1]

    # check if valid hours are greater than 16
    day_x = day_x.assign(valid_hours=(day_x['paxwwmd'] + day_x['paxswmd']) / 60)
    day_x = day_x[day_x['valid_hours'] > 16]

    # check if there are at least 4 days of data
    day_x = day_x.groupby('seqn').filter(lambda x: len(x) >= 4)

    # read all minute-level files
    min_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading minute-level files"):
        itr_x = pd.read_sas(f"{file_dir}/PAXMIN_{version}.xpt", chunksize=100000)
        for chunk in tqdm(itr_x, desc=f"Processing chunks for version {version}"):
            curr = clean_data(chunk, day_x)
            curr = curr[curr['seqn'] == person_id]
            min_x = pd.concat([min_x, curr], ignore_index=True)

    min_x = min_x.rename(columns=str.lower)
    min_x = remove_bytes(min_x)

    if verbose:
        print(f"Read {min_x.shape[0]} minute-level records for person {person_id}")

    # add header data
    head_x = pd.DataFrame()
    for version in tqdm(versions, desc="Reading header files"):
        curr = pd.read_sas(f"{file_dir}/PAXHD_{version}.xpt")
        curr = curr[curr['seqn'] == person_id]
        head_x = pd.concat([head_x, curr], ignore_index=True)

    head_x = head_x.rename(columns=str.lower)
    head_x = head_x[['seqn', 'paxftime', 'paxfday']].rename(columns={
        'paxftime': 'day1_start_time', 'paxfday': 'day1_which_day'
    })

    min_x = min_x.merge(head_x, on='seqn')
    min_x = remove_bytes(min_x)

    if verbose:
        print(f"Merged day- and minute-level data for person {person_id}")

    # calculate measure time
    min_x['measure_time'] = min_x.apply(calculate_measure_time, axis=1)
    min_x['measure_hour'] = min_x['measure_time'].dt.hour

    valid_startend = min_x.groupby(['seqn', 'paxdaym']).agg(
        start=('measure_hour', 'min'),
        end=('measure_hour', 'max')
    ).reset_index()

    min_x = min_x.merge(valid_startend, on=['seqn', 'paxdaym'])
    min_x = min_x[(min_x['start'] == 0) & (min_x['end'] == 23)]

    min_x['measure_min'] = min_x['measure_time'].dt.minute
    min_x['myepoch'] = (12 * min_x['measure_hour'] + np.floor(min_x['measure_min'] / 5 + 1)).astype(int)

    epoch = min_x[['seqn', 'paxdaym', 'myepoch']].drop_duplicates()

    check = epoch.groupby(['seqn', 'paxdaym']).size().reset_index(name='n_epoch')
    epoch2 = epoch.merge(check, on=['seqn', 'paxdaym'])
    epoch2 = epoch2[epoch2['n_epoch'] == 288]

    check2 = epoch2.groupby('seqn').size().reset_index(name='n_days')
    epoch3 = epoch2.merge(check2, on='seqn')
    epoch3 = epoch3[epoch3['n_days'] >= 4]

    min_x = min_x.rename(columns={
        'paxmxm': 'x', 'paxmym': 'y', 'paxmzm': 'z', 'measure_time': 'timestamp'
    })

    if verbose:
        print(f"Renamed columns and set timestamp index for person {person_id}")

    min_x.set_index('timestamp', inplace=True)
    min_x = min_x[['x', 'y', 'z']]
    min_x['ENMO'] = calculate_enmo(min_x)

    if verbose:
        print(f"Calculated ENMO for person {person_id}")

    return min_x




def remove_bytes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes([object]):  # Select columns with object type (likely byte strings)
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df

def clean_data(df: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    df = df[df['SEQN'].isin(days['seqn'])]
    df = df[df['PAXMTSM'] != -0.01]
    df = df[~df['PAXPREDM'].isin([3, 4])]
    df = df[df['PAXQFM'] < 1]
    return df

def calculate_measure_time(row):
    base_time = datetime.strptime(row['day1_start_time'], "%H:%M:%S")
    measure_time = base_time + timedelta(seconds=row['paxssnmp'] / 80)
    return measure_time