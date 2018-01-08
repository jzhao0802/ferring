# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:35:31 2017

@author: Shaun.Gupta
"""
from table_parsers.base_table_parser import BaseTableParser

class DispositionParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)
        #self._useful_disp_info = ['ACTIVE LABOR', 'DELIVERY', 'L&D DISCHARGE', 'COMPLETED']
        #Get Delivery info from delivery data table - no need for duplicate info (checked timestamps are identical)
        self._useful_disp_info = ['ACTIVE LABOR', 'L&D DISCHARGE', 'COMPLETED']

    def extract_disp_info(self, df):
        df.loc[df['DSTERM'] == 'COMPLETED', 'DSSTDTC'] = True
        df = df[df['DSTERM'].isin(self._useful_disp_info)][['DSTERM', self._primary_key, 'DSSTDTC']]
        return df.pivot(index=self._primary_key, columns='DSTERM', values='DSSTDTC').reset_index()\
            .rename(columns={'ACTIVE LABOR': 'ACTIVE_LABOUR_TIME', 'L&D DISCHARGE': 'DISCHARGE_TIME'})
        
    def _process_table(self, raw_df, processed_df):
        return self.extract_disp_info(processed_df)