# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:55:42 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser


class ObstetricHistoryParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)
        self._gestation_cols = set(['GADAY', 'GAWEEK'])

    def extract_pregnancy_type_info(self, df):
        return pd.get_dummies(df[df['OHTESTCD'] == 'PREGTYPE'][[self._primary_key, 'OHSTRESC']], columns=['OHSTRESC' ], prefix='PREGTYPE')
    
    def extract_gestational_info(self, df):
        df = df[df['OHTESTCD'].isin(self._gestation_cols)][[self._primary_key, 'OHSTRESN', 'OHTESTCD']]
        df = df.pivot(index=self._primary_key, columns='OHTESTCD', values='OHSTRESN').reset_index()
        df['GESTATIONAL_AGE_DAYS'] = (df['GAWEEK']*7) + df['GADAY']
        del df['GAWEEK']
        del df['GADAY']
        return df
        
              
    def _process_table(self, raw_df, processed_df):
        df_preg_type = self.extract_pregnancy_type_info(processed_df)
        df_ga = self.extract_gestational_info(processed_df)
        return pd.merge(df_preg_type, df_ga, on=self._primary_key, how='outer')
        
        
            
          
