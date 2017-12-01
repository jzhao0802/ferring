# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:12:39 2017

@author: Shaun.Gupta
"""
import pandas as pd
class BaseTableParser(object):
    
    def __init__(self, primary_key='USUBJID'):
        super().__init__()
        self._primary_key = primary_key
    
    
    def decode_byte_str_cols(self, df):
        ''' Convert all string cols to utf-8
        '''
        df_copy = pd.DataFrame.copy(df, deep=True)
        str_df = df_copy.select_dtypes(include=['object'])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        
        for col in str_df:
            df_copy[col] = str_df[col]
        return df_copy
    
    def convert_col_to_date(self, df, date_col):
        df[date_col] = pd.to_datetime(df[date_col], format='%Y')
    
    def pre_process(self, df, date_col=None):
        df = self.decode_byte_str_cols(df)
        if date_col is not None: df = self.convert_col_to_date(df, date_col)
        return df