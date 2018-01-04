# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:10:42 2017

@author: Shaun.Gupta
"""
import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
from table_parsers.medical_history_term_mapper import MedicalHistoryTermMapper
import os

class MedicalHistoryParser(BaseTableParser):
    '''Class for parsing medical history table
    '''
    def __init__(self, study_code, primary_key='USUBJID'):
    #    super().__init__(primary_key=primary_key)
        super().__init__(study_code, primary_key=primary_key, table_name='MedicalHistory')
        #self._primary_key = primary_key
        self._mapper = MedicalHistoryTermMapper()

    def map_terms(self, df, diagnosis_col):
        self._mapper.map_terms(df, diagnosis_col, self._primary_key, self._num_patients)
        if self._log_path:
            self._mapper.log_mapped_terms(os.path.join(self._log_path, '%s_mapped_terms.xlsx'%self._table_name))

    def aggregate_counts(self, df, diagnosis_col):
        df = pd.get_dummies(df, columns=[diagnosis_col]) \
                    .filter(regex=r'^%s|%s'%(diagnosis_col, self._primary_key)) \
                    .groupby(self._primary_key) \
                    .sum() 
                    
        df.columns = ['%s_COUNT'%(col) for col in df.columns]
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
    
    
    def _process_table(self, raw_df, processed_df):
        #Use MHTERM in future??
        diagnosis_col = 'MHTERM'
        self.map_terms(processed_df, diagnosis_col)
        processed_df = self._mapper.filter_terms(processed_df, diagnosis_col)
        processed_df[diagnosis_col] = processed_df[diagnosis_col].str.upper().str.replace(' ', '_')
        df_counts = self.aggregate_counts(processed_df, diagnosis_col)
        #For now, don't get date for 004...
        if self._study_code == '004': return df_counts
        df_dates = self.aggregate_dates(processed_df, diagnosis_col, 'MHSTDTC')
        return pd.merge(df_counts, df_dates, on=self._primary_key, how='outer')
                    

        