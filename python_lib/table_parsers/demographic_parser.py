# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:45:29 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
import numpy as np

class DemographicParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)
        self._race_map = {
            '303': ('RACE_PROCESSED', {'NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER': 'OTHER', 'MULTIPLE': 'OTHER',
                                     'AMERICAN_INDIAN_OR_ALASKA_NATIVE': 'OTHER'}),
            '004': ('RACE_RAW', {'BLACK': 'BLACK_OR_AFRICAN_AMERICAN', '_______.': np.nan})
        }

    def extract_demographic_data(self, df, df_raw):
        df = df[[self._primary_key, 'AGE', 'RACE', 'COUNTRY', 'SUBJID']]

        #Add raw and processed race field
        df['RACE_PROCESSED'] = df['RACE'].str.replace(' ', '_')
        del df['RACE']
        df = pd.merge(df, df_raw[['SUBJECT', 'RACE']], left_on='SUBJID', right_on='SUBJECT')
        df['RACE_RAW'] = df['RACE'].str.replace(' ', '_').str.upper()
        del df['RACE']
        del df['SUBJECT']

        #Map race to match fields between studies
        source_race_field, race_map = self._race_map[self._study_code]
        df['RACE'] = df[source_race_field]
        for race, mapped_race in race_map.items():
            df.loc[df['RACE'] == race, 'RACE'] = mapped_race

        df['RACE_dummy'] = df['RACE']
        df['COUNTRY_dummy'] = df['COUNTRY']
        df = pd.get_dummies(df, columns=['RACE_dummy', 'COUNTRY_dummy'])
        return df
    
    def extract_vital_stats(self, raw_df):
        df_vs = raw_df[['SUBJECT', 'VSORRES1', 'VSORRES2', 'VSORRES3']] \
         .rename(columns={'VSORRES1': 'WEIGHT', 'VSORRES2': 'HEIGHT', 'VSORRES3': 'BMI', 
                       'SUBJECT': 'SUBJID'})
        return df_vs
              
    def _process_table(self, raw_df, processed_df):
        df_dm = self.extract_demographic_data(processed_df, raw_df)
        if self._study_code == '303':
            df_vs = self.extract_vital_stats(raw_df)
            print (df_vs.keys(), df_dm.keys())
            df_dm = pd.merge(df_dm, df_vs, on='SUBJID', how='outer')
            del df_dm['SUBJID']
        return df_dm
            
          
