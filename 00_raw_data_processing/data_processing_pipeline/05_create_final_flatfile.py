# -*- coding: utf-8 -*-
import pandas as pd
import set_lib_paths
from functools import reduce

import os
import numpy as np

def main(data_dir):

    df = pd.DataFrame.from_csv(os.path.join('%s/merged_data/MERGED_CM_MH_FLATFILE.csv' % (data_dir)))

    #Filter out patients  that did not complete study...a2
    df = df[df['COMPLETED'] & ~df['COMPLETED'].isnull()]
    #Add gestational age in weeks
    #df['GESTATIONAL_AGE_WEEKS'] = df['GESTATIONAL_AGE_DAYS']/7

    #Combine vaginal modes of delivery
    df.loc[df['MODE_OF_DELIVERY'].str.contains('VAGINAL'), 'MODE_OF_DELIVERY'] = 'VAGINAL'


    #Add time to oxytocin administration
    df['TIME_DELTA_EX_START_OXYTOCIN_ADMIN'] = (pd.to_datetime(df['OXYTOCIN_ADMINISTRATION_TIME']) - pd.to_datetime(
        df['EX_START_TIME'])) / np.timedelta64(1, 'h')

    #Add time to onset of labour
    df['TIME_DELTA_EX_START_ONSET_LABOUR'] = (pd.to_datetime(df['ACTIVE_LABOUR_TIME']) - pd.to_datetime(
        df['EX_START_TIME'])) / np.timedelta64(1, 'h')

    #Create label column using outcome definition
    df['LABEL'] = (df['TIME_DELTA_EX_START_ONSET_LABOUR'] <=24) & (df['MODE_OF_DELIVERY'] == 'VAGINAL')
    df['LABEL'] = df['LABEL'].astype(int)

    df = df.filter(regex='(?=^((?!Unnamed).)*$)')

    #Remove nan values from canada dummy flag
    df['COUNTRY_dummy_CAN'][df['COUNTRY_dummy_CAN'].isnull()] = 0

    #Write out flatfile
    pd.DataFrame.to_csv(df, os.path.join('%s/merged_data/PROCESSED_FLATFILE.csv' % (data_dir)))

if __name__ == '__main__':
    output_base_dir = 'F:/Projects/Ferring/data/pre_modelling/'

    main(output_base_dir)
