# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:55:42 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser


class VitalStatsParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)


    def extract_vital_stats(self, df):
        df_vs = df[[self._primary_key, 'VSTESTCD', 'VSSTRESN']] \
        .pivot(index=self._primary_key, columns='VSTESTCD', values='VSSTRESN') \
        .reset_index()
        df_vs['BMI'] = df_vs['WEIGHT']/((df_vs['HEIGHT']/100)**2)
        return df_vs
              
    def _process_table(self, raw_df, processed_df):
        return self.extract_vital_stats(processed_df)
            
          
