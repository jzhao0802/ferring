# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:29:22 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
from functools import partial


class ExposureParser(BaseTableParser):
    
    def __init__(self, study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)
        
    def extract_insertion_removal_times(self, df):
        print(df.keys())
        return df[[self._primary_key, 'EXSTDTC', 'EXENDTC']].rename(columns={
                'EXSTDTC':'EX_START_TIME', 'EXENDTC': 'EX_END_TIME'})
    
    
    def _process_table(self, raw_df, processed_df):
        return self.extract_insertion_removal_times(processed_df)