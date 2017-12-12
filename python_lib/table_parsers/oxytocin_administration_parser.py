# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:55:42 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser


class OxytocinAdministrationParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)

    def extract_oxytocin_administration_info(self, df):
        df = df[[self._primary_key, 'OADOSTOT']]
        df['OXYTOCIN_ADMINISTERED'] = ~df['OADOSTOT'].isnull()
        return df.rename(columns={'OADOSTOT': 'OXYTOCIN_DOSAGE'})
    
    def _process_table(self, raw_df, processed_df):
        return self.extract_oxytocin_administration_info(processed_df)
        
        
            
          
