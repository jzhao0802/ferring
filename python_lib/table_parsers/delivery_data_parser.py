# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:26:56 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
from functools import partial


class DeliveryDataParser(BaseTableParser):
    
    def __init__(self, study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)
        self._delivery_methods = set(['MODE OF DELIVERY', 'Mode Of Delivery'])

    def extract_delivery_time(self, df):
        return df[['USUBJID', 'DDDTC']].rename(columns={'DDDTC': 'DD_DELIVERY_TIME'})

    def extract_delivery_mode_information(self, df):
        #return df[df['DDTEST'].isin(self._delivery_methods)][[self._primary_key, 'DDSTRESC', 'DDDTC']] \
        #    .rename(columns={'DDSTRESC': 'DD_DELIVERY_METHOD', 'DDDTC': 'DD_DELIVERY_TIME'})
        df['DDTEST'] = df['DDTEST'].str.upper().str.replace(' ', '_')
        return df.pivot(index='USUBJID', columns='DDTEST', values='DDSTRESN').reset_index()


    def _process_table(self, raw_df, processed_df):
        df_dd = self.extract_delivery_mode_information(processed_df)
        df_dt = self.extract_delivery_time(processed_df)
        return pd.merge(df_dd, df_dt, on='USUBJID', how='outer')