# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:45:29 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser


class DemographicParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)


    def extract_demographic_data(self, df):
        df = df[[self._primary_key, 'AGE', 'RACE', 'COUNTRY', 'SUBJID']]
        df['RACE'] = df['RACE'].str.replace(' ', '_')
        df = pd.get_dummies(df, columns=['RACE', 'COUNTRY'])
        return df
    
    def extract_vital_stats(self, raw_df):
        df_vs = raw_df[['SUBJECT', 'VSORRES1', 'VSORRES2', 'VSORRES3']] \
         .rename(columns={'VSORRES1': 'WEIGHT', 'VSORRES2': 'HEIGHT', 'VSORRES3': 'BMI', 
                       'SUBJECT': 'SUBJID'})
        return df_vs
              
    def _process_table(self, raw_df, processed_df):
        df_dm = self.extract_demographic_data(processed_df)
        if self._study_code == '303':
            df_vs = self.extract_vital_stats(raw_df)
            print (df_vs.keys(), df_dm.keys())
            df_dm = pd.merge(df_dm, df_vs, on='SUBJID')
            del df_dm['SUBJID']
        return df_dm
            
          
