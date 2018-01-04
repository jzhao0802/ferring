# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:12:39 2017

@author: Shaun.Gupta
"""
import pandas as pd
from abc import ABCMeta, abstractmethod, ABC


class BaseTableParser(ABC):
    ''' Abstract Class that contains common functions useful for parsing all tables.
    '''    
      
    def __init__(self, study_code, primary_key='USUBJID', table_name=''):
        super().__init__()
        self._primary_key = primary_key
        self._study_code = study_code
        self._table_name = table_name

    def set_log_path(self, path):
        self._log_path = path

    def set_num_patients(self, num_patients):
        self._num_patients = num_patients

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
        if df is None: return None
        df = self.decode_byte_str_cols(df)
        if date_col is not None: df = self.convert_col_to_date(df, date_col)
        return df
    
    @abstractmethod   
    def _process_table(self, *args, **kwargs):
        NotImplementedError('This method must be implemented in all  \
                            child classes!')
    
    def process_table(self, *args, **kwargs):
        print ('Running ' + str(self))
        if args[0] is None and args[1] is None: return None
        if kwargs.get('pre_process', False) : 
            raw_df = self.pre_process(args[0])
            processed_df = self.pre_process(args[1])
            args = tuple([raw_df, processed_df] + [arg for arg in args[2:]])
        if 'pre_process' in kwargs: kwargs.pop('pre_process')
        return self._process_table(*args, **kwargs)