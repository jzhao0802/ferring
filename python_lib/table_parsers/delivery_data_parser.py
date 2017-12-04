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
        
        
    def extract_delivery_mode_information(self, df):
        return df[df['DDTEST'] == 'MODE OF DELIVERY'][[self._primary_key, 'DDSTRESC', 'DDDTC']] \
            .rename(columns={'DDSTRESC': 'DD_DELIVERY_METHOD', 'DDDTC': 'DD_DELIVERY_TIME'})

    def _process_table(self, raw_df, processed_df, delivery_type_col='DDTEST'):
        return self.extract_delivery_mode_information(processed_df)