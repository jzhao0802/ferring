# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:52:49 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
from functools import partial


class BishopScoreParser(BaseTableParser):
    
    def __init__(self,  study_code, primary_key='USUBJID'):
        super().__init__(study_code, primary_key=primary_key)
        self._time_map = {
                    1: 'BASELINE',
                    2: '6',
                    3: '12',
                    4: '18',
                    5: '24'
                }

    def extract_bishop_scores(self, df):
        df = df[[self._primary_key, 'BSSEQ', 'BSSTRESN']]
        #df['BSTPT'] = df['BSTPT'].str.split().apply(lambda x: 'BS_' + x[0])
        df['BSTPT'] = df['BSSEQ'].apply(lambda x: 'BS_' + self._time_map[x])
        df = df.pivot(index=self._primary_key, columns='BSTPT', values='BSSTRESN')
        return df.reset_index()
    
    def _process_table(self, raw_df, processed_df):
        return self.extract_bishop_scores(processed_df)