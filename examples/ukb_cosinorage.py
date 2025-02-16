import pandas as pd
import numpy as np
import cosinorage
from cosinorage.datahandlers import UKBDataHandler
from cosinorage.bioages import CosinorAge
from tqdm import tqdm
def get_chrono_age_and_gender(file_path, eid):
    data = pd.read_csv(file_path)
    data = data[data['eid'] == eid]
    age = data['age'].values[0]
    if data['sex'].values[0] == 'Male':
        gender = 'male'
    elif data['sex'].values[0] == 'Female':
        gender = 'female'
    else:
        gender = 'unknown'
    return int(age), gender

def get_gt_cosinor_age(file_path, eid):
    data = pd.read_csv(file_path)
    data = data[data['eid'] == eid]
    gt_cosinor_age = data['coisnor_age'].values[0]
    return float(gt_cosinor_age)

def main():
    df = pd.read_csv('../data/ukb/cosinorage_param.csv')
    qa_file_path = '../data/ukb/UKB Acc Quality Control.csv'
    enmo_file_dir = '../data/ukb/UKB Sample Data/1_raw5sec_long'

    # iterate over each row in the dataframe with tqdm
    for index, row in tqdm(df.iterrows()):
        eid = row['eid']
        gt_mesor = row['mesor']
        gt_amplitude = row['amp1']
        gt_acrophase = row['phi1']
        gt_cosinor_age = row['cosinoage']
        gt_cosinor_age_advance = row['cosinoage_advance']

        ukb_data = UKBDataHandler(qa_file_path=qa_file_path, ukb_file_dir=enmo_file_dir, eid=eid, verbose=False)

        record = [
            {'handler': ukb_data, 
            'age': get_chrono_age_and_gender('../data/Age_sex_data/ukb_age_sex.csv', eid)[0], 
            'gender': get_chrono_age_and_gender('../data/Age_sex_data/ukb_age_sex.csv', eid)[1], 
            'gt_cosinor_age': get_gt_cosinor_age('../data/Age_sex_data/ukb_cosinor_age.csv', eid)    
            }
        ]

        cosinor_age = CosinorAge(records=record)
        pred_mesor = cosinor_age.get_predictions()[0]['mesor']
        pred_amplitude = cosinor_age.get_predictions()[0]['amp1']
        pred_acrophase = cosinor_age.get_predictions()[0]['phi1']
        pred_cosinor_age = cosinor_age.get_predictions()[0]['cosinoage']
        pred_cosinor_age_advance = cosinor_age.get_predictions()[0]['cosinoage_advance']

        age = record[0]['age']
        gender = record[0]['gender']

        # save all the values to a csv
        with open('../data/ukb/cosinor_age_predictions.csv', 'a') as f:
            f.write(f"{eid},{age},{gender},{gt_mesor},{gt_amplitude},{gt_acrophase},{pred_mesor},{pred_amplitude},{pred_acrophase},{gt_cosinor_age},{pred_cosinor_age},{gt_cosinor_age_advance},{pred_cosinor_age_advance}\n")

if __name__ == "__main__":
    main()