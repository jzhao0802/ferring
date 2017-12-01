# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:52:49 2017

@author: Shaun.Gupta
"""

import pandas as pd
from table_parsers.base_table_parser import BaseTableParser
from functools import partial


class BishopScoreParser(BaseTableParser):
    
    def __init__(self, primary_key='USUBJID'):
        super().__init__(primary_key=primary_key)


    def extract_bishop_scores(self, df):
        
    
    def _process_table(self, df):
        return self.extract_bishop_scores(df)