# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:49:27 2017

@author: Shaun.Gupta
"""

from table_parsers.medical_history_parser import MedicalHistoryParser
import pandas as pd

class AdverseEventsParser(MedicalHistoryParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)

    def extract_adverse_events(self, df, exposure_table):
        exposure_table['EX_START_TIME'] = pd.to_datetime(exposure_table['EX_START_TIME'])

        df['AESTDTC'] = pd.to_datetime(df['AESTDTC'])
        df = pd.merge(df, exposure_table, on=self._primary_key, how='left')

        df = df[df['AESTDTC'] <= df['EX_START_TIME']]
        print(df)

        diagnosis_col = 'AETERM'
        df[diagnosis_col] = df[diagnosis_col].str.upper().str.replace(' ', '_')
        df_counts = self.aggregate_counts(df, diagnosis_col)
        #For now, don't get date for 004...
        if self._study_code == '004': return df_counts
        df_dates = self.aggregate_dates(df, diagnosis_col, 'AESTDTC')
        return pd.merge(df_counts, df_dates, on=self._primary_key, how='outer')
        
    def _process_table(self, raw_df, processed_df, extra_tables={}):
        return self.extract_adverse_events(processed_df, extra_tables['ex'])