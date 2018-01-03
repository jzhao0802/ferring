# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:49:27 2017

@author: Shaun.Gupta
"""

from table_parsers.medical_history_parser import MedicalHistoryParser
import pandas as pd

class ConcomitantMedicationParser(MedicalHistoryParser):

    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)

    def extract_cm_events(self, df, exposure_table):
        exposure_table['EX_START_TIME'] = pd.to_datetime(exposure_table['EX_START_TIME'])

        df['CMSTDTC'] = pd.to_datetime(df['CMSTDTC'])
        df = pd.merge(df, exposure_table, on=self._primary_key, how='left')

        df = df[df['CMSTDTC'].dt.date < df['EX_START_TIME'].dt.date]
        #print(df[['EX_START_TIME', 'CMSTDTC']])
        print(df)
        print(len(df))
        print (len(df['USUBJID'].unique()))
        diagnosis_col = 'CMTRT'
        df[diagnosis_col] = df[diagnosis_col].str.upper().str.replace(' ', '_')
        df_counts = self.aggregate_counts(df, diagnosis_col)
        df_dates = self.aggregate_dates(df, diagnosis_col, 'CMSTDTC')
        return pd.merge(df_counts, df_dates, on=self._primary_key, how='outer')

    def _process_table(self, raw_df, processed_df, extra_tables={}):
        return self.extract_cm_events(processed_df, extra_tables['ex'])