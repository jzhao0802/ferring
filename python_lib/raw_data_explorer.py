# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:51:35 2017

@author: Shaun.Gupta
"""

import pandas as pd
import os
import itertools

class RawDataExplorer(object):
    
    def __init__(self):
        pass
    
    def set_base_dir(self, base_dir):
        self._base_dir = base_dir
        
    def read_file(self, filename):
        try:
            return pd.read_sas(filename)
        except:
            return None

    def load_data(self):
        self._data = {filename:self.read_file(os.path.join(self._base_dir, filename)) for
                      filename in os.listdir(self._base_dir)}
        
    def get_data(self, filename=None, default=None):
        if filename: return self._data.get(filename, default)
        return self._data
    
    def get_none_dfs(self):
        return [(f, df) for f,df in self._data.items() if df is None]    
    
    def get_all_vars_by_df(self):
        return [set(df.keys()) for f,df in self._data.items() if df is not None]
    
    def get_common_vars(self):
        return set.intersection(*self.get_all_vars_by_df())
    
    def get_all_vars(self):
        return set.union(*self.get_all_vars_by_df())
    
    def get_unique_vals(self, common=False):
        if common: variables = self.get_common_vars()
        else: variables = self.get_all_vars()
        return {key : set(list(itertools.chain(*[df.get(key, pd.Series()).unique().tolist() for df in list(self._data.values()) if df is not None]))) for key in variables}
    
    def count_unique_vals(self, variable_name=''):
        vals = self.get_unique_vals()
        if variable_name: return len(vals[variable_name])
        else: return {var:len(val) for var,val in vals.items()}
    
    def count_rows_per_patient(self, id_col='USUBJID'):
        return {f:df.groupby('USUBJID').size().reset_index(name='counts') for f,df in self._data.items() if df is not None}
    
    def get_avg_rows_per_patient(self, id_col='USUBJID'):
        patient_ids = {}
        for df in list(self.count_rows_per_patient().values()):
            d = df.to_dict('records')
            for r in d:
                patient_ids[r['USUBJID']] = patient_ids.get(r['USUBJID'], 0)+r['counts']
        df = pd.DataFrame.from_dict(patient_ids, orient='index')
        return df[0].mean()
    
    def get_tables_with_min_n_rows(self, n_min_rows):
        return {f:df for f,df in self._data.items() if df is not None and len(df) >= n_min_rows }
    
    def get_n_rows_all_tables(self):
        return {f:len(df) for f,df in self._data.items() if df is not None}
    #def count_rows_per_patient(self):
        
        
if __name__ == '__main__':
    dirs = ['F:/Projects/Ferring/data/raw_data/DK0-Propess Predictability project/Miso-Obs-004/sdtm_data_Miso_Obs_004/', 'F:/Projects/Ferring/data/raw_data/DK0-Propess Predictability project/Miso-Obs-303/sdtm_data_Miso_Obs_303/']
    #Check it is possible to open all files in directory
    for i in dirs:
        [print (pd.read_sas(os.path.join(i,f))) for  f in os.listdir(i)]
        
        
    #Check keys for single dir