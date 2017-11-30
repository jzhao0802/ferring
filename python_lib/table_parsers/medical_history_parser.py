# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:10:42 2017

@author: Shaun.Gupta
"""
import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
from functools import partial

class MedicalHistoryParser(BaseTableParser):
    
    def __init__(self, primary_key='USUBJID'):
    #    super().__init__(primary_key=primary_key)
        super().__init__()
        self._primary_key = primary_key
    
    def aggregate_counts(self, df, diagnosis_col):
        df = pd.get_dummies(df, columns=[diagnosis_col]) \
                    .filter(regex=r'^%s|%s'%(diagnosis_col, self._primary_key)) \
                    .groupby(self._primary_key) \
                    .sum() 
                    
        df.columns = ['%s_COUNT'%(col.upper().replace(' ', '_')) for col in df.columns]
        return df.reset_index()
                    
    def aggregate_dates(self, df, diagnosis_col, date_col):
        df_grouped = df[[self._primary_key, diagnosis_col, date_col]].sort_values(date_col) \
                .groupby([self._primary_key, diagnosis_col])
        
        
        df_first_date = df_grouped.first()[date_col].reset_index()
        df_first_date[diagnosis_col] = diagnosis_col + '_' +  df_first_date[diagnosis_col].str.upper().str.replace(' ', '_') + '_FIRST_EXPDT'
        df_first_date = df_first_date.pivot(index=self._primary_key, columns=diagnosis_col, values=date_col)
        df_first_date = df_first_date.reset_index()
        #del df_first_date[0]
                
        df_last_date = df_grouped.last()[date_col].reset_index()
        df_last_date[diagnosis_col] = diagnosis_col + '_' +  df_last_date[diagnosis_col].str.upper().str.replace(' ', '_') + '_LAST_EXPDT'
        df_last_date = df_last_date.pivot(index=self._primary_key, columns=diagnosis_col, values=date_col)
        df_last_date = df_last_date.reset_index()
        #del df_last_date[0]
            
        
        return pd.merge(df_first_date, df_last_date, on=self._primary_key)
    
    
    def process_table(self, df, diagnosis_col='MHDECOD', date_col='MHSTDTC', pre_process=False):
        if pre_process: df = self.pre_process(df)
        df_counts = self.aggregate_counts(df, diagnosis_col)
        df_dates = self.aggregate_dates(df, diagnosis_col, date_col)
        return pd.merge(df_counts, df_dates, on=self._primary_key)
                    

        